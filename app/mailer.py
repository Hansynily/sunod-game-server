from __future__ import annotations

from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import parseaddr
import json
import urllib.error
import urllib.request
from functools import lru_cache
import logging
import os
from pathlib import Path
import smtplib
import ssl
from urllib.parse import quote_plus

from jinja2 import Environment, FileSystemLoader, select_autoescape


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
_EMAIL_ENVIRONMENT = Environment(
    loader=FileSystemLoader(str(ROOT / "templates")),
    autoescape=select_autoescape(["html", "xml"]),
)


@dataclass(frozen=True, slots=True)
class MailerSettings:
    host: str | None
    port: int
    username: str | None
    password: str | None
    from_email: str | None
    app_public_url: str
    verification_ttl_hours: int
    use_starttls: bool
    use_ssl: bool
    brevo_api_key: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.host and self.from_email)


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    mode: str
    message: str
    verification_link: str | None = None


@lru_cache(maxsize=1)
def get_mailer_settings() -> MailerSettings:
    port_raw = (os.getenv("SMTP_PORT") or "587").strip()
    ttl_raw = (os.getenv("EMAIL_VERIFICATION_TTL_HOURS") or "24").strip()

    try:
        port = int(port_raw)
    except ValueError as exc:
        raise RuntimeError("SMTP_PORT must be an integer.") from exc

    try:
        verification_ttl_hours = int(ttl_raw)
    except ValueError as exc:
        raise RuntimeError("EMAIL_VERIFICATION_TTL_HOURS must be an integer.") from exc

    return MailerSettings(
        host=(os.getenv("SMTP_HOST") or "").strip() or None,
        port=port,
        username=(os.getenv("SMTP_USERNAME") or "").strip() or None,
        password=(os.getenv("SMTP_PASSWORD") or "").strip() or None,
        from_email=(os.getenv("SMTP_FROM_EMAIL") or "").strip() or None,
        app_public_url=(os.getenv("APP_PUBLIC_URL") or "http://127.0.0.1:8000").strip().rstrip("/"),
        verification_ttl_hours=max(1, verification_ttl_hours),
        use_starttls=_env_flag("SMTP_STARTTLS", default=True),
        use_ssl=_env_flag("SMTP_SSL", default=False),
        brevo_api_key=(os.getenv("BREVO_API_KEY") or "").strip() or None,
    )


def build_verification_link(token: str) -> str:
    settings = get_mailer_settings()
    return f"{settings.app_public_url}/verify-email?token={quote_plus(token)}"


def render_verification_email(*, username: str, verification_link: str, expires_in_hours: int) -> tuple[str, str]:
    html_template = _EMAIL_ENVIRONMENT.get_template("email_verification_email.html")
    html_body = html_template.render(
        username=username,
        verification_link=verification_link,
        expires_in_hours=expires_in_hours,
    )
    text_body = (
        f"Hello {username},\n\n"
        "Your SUNOD account is ready for email verification.\n"
        f"Open this link within {expires_in_hours} hours:\n{verification_link}\n\n"
        "If you did not request this account, you can ignore this email.\n"
    )
    return html_body, text_body


def send_email(
    *,
    to_email: str,
    subject: str,
    html_body: str,
    text_body: str,
    verification_link: str | None = None,
) -> DeliveryResult:
    settings = get_mailer_settings()
    # Brevo's HTTPS API first when a key is set: hosts like Render's free tier block
    # outbound SMTP ports, but not HTTPS. Without a key, SMTP below is used as before.
    if settings.brevo_api_key and settings.from_email:
        _send_via_brevo(settings, to_email=to_email, subject=subject, html_body=html_body, text_body=text_body)
        return DeliveryResult(
            mode="brevo",
            message="Verification email sent successfully.",
        )

    if not settings.enabled:
        LOGGER.info(
            "SMTP is not configured. Verification email was not sent to %s. Subject=%s",
            to_email,
            subject,
        )
        LOGGER.info("Verification email body:\n%s", text_body)
        return DeliveryResult(
            mode="log_only",
            message="SMTP not configured. Use the verification link below.",
            verification_link=verification_link,
        )

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.from_email
    message["To"] = to_email
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    if settings.use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(settings.host, settings.port, context=context, timeout=20) as client:
            _login_and_send(client, settings, message)
    else:
        with smtplib.SMTP(settings.host, settings.port, timeout=20) as client:
            client.ehlo()
            if settings.use_starttls:
                client.starttls(context=ssl.create_default_context())
                client.ehlo()
            _login_and_send(client, settings, message)

    return DeliveryResult(
        mode="smtp",
        message="Verification email sent successfully.",
    )


BREVO_SEND_URL = "https://api.brevo.com/v3/smtp/email"


def _send_via_brevo(settings: MailerSettings, *, to_email: str, subject: str, html_body: str, text_body: str) -> None:
    sender_name, sender_email = parseaddr(settings.from_email or "")
    payload = {
        "sender": {"name": sender_name or "SUNOD", "email": sender_email or settings.from_email},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": html_body,
        "textContent": text_body,
    }
    request = urllib.request.Request(
        BREVO_SEND_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "api-key": settings.brevo_api_key or "",
            "content-type": "application/json",
            "accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if response.status >= 300:
                raise RuntimeError(f"Brevo send failed: HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Brevo send failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Brevo send failed: {exc.reason}") from exc


def _login_and_send(client: smtplib.SMTP, settings: MailerSettings, message: EmailMessage) -> None:
    if settings.username:
        client.login(settings.username, settings.password or "")
    client.send_message(message)


def _env_flag(name: str, *, default: bool) -> bool:
    raw_value = (os.getenv(name) or "").strip().lower()
    if not raw_value:
        return default
    return raw_value not in {"0", "false", "no", "off"}

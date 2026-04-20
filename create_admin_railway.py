import os

def get_input(env_key, prompt_text):
    value = os.getenv(env_key)
    if value:
        return value.strip()
    return input(prompt_text).strip()

def main():
    try:
        username = get_input("ADMIN_USERNAME", "Admin username: ")
        password = get_input("ADMIN_PASSWORD", "Admin password: ")

        if not username or not password:
            raise ValueError("Username and password cannot be empty")

        print(f"Creating admin user: {username}")

        # === YOUR ACTUAL LOGIC HERE ===
        # Example placeholder:
        # create_admin_user(username, password)

        print("Admin created successfully")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
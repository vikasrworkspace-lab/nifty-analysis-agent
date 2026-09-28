import os
import sys
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

def main():
    load_dotenv()
    client_id = os.getenv("FYERS_APP_ID")
    secret_key = os.getenv("FYERS_SECRET_KEY")
    redirect_uri = os.getenv("FYERS_REDIRECT_URI")
    
    if not client_id or not secret_key:
        print("Error: FYERS_APP_ID or FYERS_SECRET_KEY not found in .env")
        sys.exit(1)
        
    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code"
    )
    
    # 1. Generate Auth URL
    response = session.generate_authcode()
    print("=" * 60)
    print("ACTION REQUIRED: Open this URL in your browser to login:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    print("\nAfter logging in, you will be redirected to an error/blank page.")
    print("Look at the URL in your browser. It will look something like:")
    # Derive the example from the configured redirect so it cannot go stale.
    hint = (redirect_uri or "<FYERS_REDIRECT_URI is not set>").rstrip("/")
    print(f"{hint}/?auth_code=SOME_LONG_CODE&state=None\n")
    
    # 2. Get auth code
    auth_code_input = input("Paste the ENTIRE redirected URL here (or just the auth_code): ").strip()
    
    if "auth_code=" in auth_code_input:
        # Extract just the code from the URL
        auth_code = auth_code_input.split("auth_code=")[1].split("&")[0]
    else:
        auth_code = auth_code_input
        
    if not auth_code:
        print("Invalid auth code.")
        sys.exit(1)
        
    # 3. Generate Access Token
    session.set_token(auth_code)
    try:
        response = session.generate_token()
        access_token = response['access_token']
        
        # Save to file
        with open(".fyers_token", "w") as f:
            f.write(access_token)
            
        print("\nSUCCESS! Access token generated and saved to .fyers_token")
        
    except Exception as e:
        print(f"\nERROR generating token: {e}")
        print("Please check if your App ID and Secret Key in .env are exactly correct.")

if __name__ == "__main__":
    main()

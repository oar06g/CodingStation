import hashlib
import hmac
import datetime

def generate_secret_key(secret_key: str, timestamp: int) -> str:
  # Convert the timestamp to bytes
  timestamp_bytes = str(timestamp).encode('utf-8')
  
  # Create a new HMAC object using the secret key and the timestamp
  hmac_object = hmac.new(secret_key.encode('utf-8'), timestamp_bytes, hashlib.sha256)
  
  # Generate the HMAC digest and return it as a hexadecimal string
  return hmac_object.hexdigest()
if __name__ == "__main__":
  secret = "my_secret_key"
  timestamp = 1780690739
  
  secret_key = generate_secret_key(secret, timestamp)
  print(f"Generated secret key: {secret_key}")

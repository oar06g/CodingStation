import hashlib
import hmac, os
import datetime

class Storage:
  def __init__(self):
    self._secrets = {}
  
  def store(self, key, secret) -> bool:
    self._secrets[key] = secret
    return True
  
  def get(self, key) -> str:
    return self._secrets.get(key, None)
  
  def all(self) -> dict:
    return self._secrets
  
class SecretGenerator:
  def __init__(self, storage: Storage):
    self._storage = storage

  def generate(self, secret, timestamp = None) -> str:
    if timestamp is None:
      timestamp = datetime.datetime.now().isoformat()
    print(f"Generating secret with timestamp: {timestamp}")
    key = hashlib.sha256((secret + timestamp).encode()).hexdigest()
    self._storage.store(key, secret)
    return key

manager = SecretGenerator(Storage())

while True:
  command = input("$ ")
  if command == "exit":
    break
  if command == "list":
    print(manager._storage.all())
    continue
  if command == "":
    continue
  if command.startswith("get "):
    key = command[4:]
    print(manager._storage.get(key))
    continue
  if command.startswith("gen "):
    command = command[4:]
    print(manager.generate(command))
  if command == 'all':
    print(manager._storage.all())
  if command == 'clear':
    os.system('cls')
import os
import json
import redis
from typing import Dict, Any

class Storage:
    def __init__(self, redis_url=None, fallback_file='storage.json'):
        self.redis_url = redis_url or os.getenv('REDIS_URL', None)
        self.fallback_file = fallback_file
        self.redis = redis.Redis.from_url(self.redis_url) if self.redis_url else None
        
    def set(self, key: str, value: Dict[str, Any]):
        if self.redis:
            self.redis.set(key, json.dumps(value))
        else:
            data = self._load_file()
            data[key] = value
            with open(self.fallback_file, 'w') as f:
                json.dump(data, f)
    
    def get(self, key: str):
        if self.redis:
            val = self.redis.get(key)
            return json.loads(val) if val else None
        else:
            data = self._load_file()
            return data.get(key, None)
    
    def _load_file(self):
        if os.path.exists(self.fallback_file):
            with open(self.fallback_file, 'r') as f:
                return json.load(f)
        return {}

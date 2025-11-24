# test_redis_client.py
import redis

r = redis.Redis(host="localhost", port=6379)
print("Ping:", r.ping())
print("Modules:", r.execute_command("MODULE", "LIST"))
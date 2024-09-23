import os
import redis

# this establishes the connection to the redis server;
client = redis.Redis.from_url(
    os.environ["REDIS_URI"],
    decode_responses=True #redis sends db info by default as a series of bytes; this auto decodes these bytes into strings
)
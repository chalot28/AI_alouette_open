import random
from datetime import datetime, timedelta

random.seed(123)

levels = ["INFO", "DEBUG", "WARN", "TRACE"]

services = [
    "UserService",
    "PaymentGateway",
    "OrderService",
    "AuthService",
    "DBConnector",
    "CacheService",
    "InventoryAPI",
    "NotificationService",
    "ShippingService",
    "RecommendationEngine",
    "SearchService",
    "AnalyticsService",
    "FileUploadService",
    "SchedulerService",
    "WebhookHandler",
    "ConfigService",
    "RatingService",
    "CouponService",
    "ChatService",
    "BackupService",
]

base_time = datetime(2026, 5, 28, 0, 0, 0)

info_messages = [
    lambda: (
        f"User login successful userId={random.randint(1000, 99999)} ip={random.randint(10, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    ),
    lambda: (
        f"Order created successfully orderId=ord_{random.randint(1000, 99999)} userId={random.randint(1000, 99999)} total={random.randint(10, 9999)}"
    ),
    lambda: (
        f"Payment processed transactionId=tx_{random.randint(1000, 99999)} amount={random.randint(100, 99999)} status=success"
    ),
    lambda: (
        f"Cache hit for key=user:{random.randint(1000, 99999)} latency={random.randint(1, 10)}ms"
    ),
    lambda: (
        f"DB query completed in {random.randint(1, 100)}ms query=SELECT * FROM orders WHERE id={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Shipment created shipmentId=shp_{random.randint(1000, 9999)} carrier={random.choice(['GHN', 'GHTK', 'VNPost', 'Viettel'])}"
    ),
    lambda: (
        f"Email sent to user={random.randint(1000, 99999)} template={random.choice(['welcome', 'order_confirmation', 'reset_password'])}"
    ),
    lambda: (
        f"API response successful endpoint={random.choice(['/api/users', '/api/orders', '/api/products'])} status=200 duration={random.randint(10, 500)}ms"
    ),
    lambda: (
        f"Scheduled job completed jobId=job_{random.randint(1000, 9999)} type={random.choice(['cleanup', 'sync', 'backup', 'report'])}"
    ),
    lambda: (
        f"File uploaded successfully userId={random.randint(1000, 99999)} fileSize={random.randint(1024, 1048576)}bytes type={random.choice(['image', 'document', 'video'])}"
    ),
    lambda: (
        f"Product inventory updated productId=prod_{random.randint(1000, 9999)} quantity={random.randint(10, 1000)} warehouse=WH-{random.randint(1, 10)}"
    ),
    lambda: (
        f"Webhook delivered successfully hookId=hook_{random.randint(1000, 9999)} event={random.choice(['order.placed', 'payment.received', 'user.registered'])}"
    ),
    lambda: (
        f"Cache refreshed for key=product:{random.randint(1000, 9999)} TTL={random.randint(300, 3600)}s"
    ),
    lambda: (
        f"Search results returned query={random.choice(['laptop', 'phone', 'book', 'shoe'])} hits={random.randint(1, 100)} time={random.randint(5, 200)}ms"
    ),
    lambda: (
        f"Rate limit status normal userId={random.randint(1000, 99999)} remaining={random.randint(50, 1000)} window=1m"
    ),
    lambda: (
        f"Authentication token refreshed userId={random.randint(1000, 99999)} expiresIn={random.randint(3600, 86400)}s"
    ),
    lambda: (
        f"Notification pushed to device deviceId=dev_{random.randint(1000, 9999)} platform={random.choice(['ios', 'android', 'web'])}"
    ),
    lambda: (
        f"Connection pool healthy pool={random.choice(['read', 'write'])} active={random.randint(1, 30)} idle={random.randint(5, 50)}"
    ),
    lambda: (
        f"gRPC health check passed service={random.choice(['user-service', 'payment-service', 'order-service'])} status=SERVING"
    ),
    lambda: (
        f"Backup completed successfully type={random.choice(['full', 'incremental', 'differential'])} size={random.randint(100, 10000)}MB duration={random.randint(60, 3600)}s"
    ),
]

debug_messages = [
    lambda: (
        f"Processing request {random.choice(['GET', 'POST', 'PUT', 'DELETE'])} /api/v{random.randint(1, 3)}/{random.choice(['users', 'orders', 'products'])}/{random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Deserializing payload for orderId=ord_{random.randint(1000, 99999)} format={random.choice(['json', 'protobuf', 'avro'])}"
    ),
    lambda: (
        f"Thread pool stats active={random.randint(1, 20)} idle={random.randint(5, 30)} queue={random.randint(0, 50)} pool={random.choice(['async', 'io', 'batch'])}"
    ),
    lambda: (
        f"Redis pipeline execution keys={random.randint(1, 100)} commands={random.choice(['GET', 'MGET', 'HGETALL', 'LRANGE'])}"
    ),
    lambda: (
        f"SQL bind params for query_{random.randint(100, 999)}: userId={random.randint(1000, 99999)}, status={random.choice(['ACTIVE', 'PENDING', 'COMPLETED'])}"
    ),
    lambda: (
        f"Cache eviction running currentSize={random.randint(1000, 10000)} maxSize={random.randint(5000, 20000)} policy={random.choice(['LRU', 'LFU', 'TTL'])}"
    ),
    lambda: (
        f"JWT token decode attempt userId={random.randint(1000, 99999)} algorithm={random.choice(['HS256', 'RS256', 'ES256'])}"
    ),
    lambda: (
        f"Circuit breaker state service={random.choice(['payment', 'inventory', 'shipping', 'notification'])} state={random.choice(['CLOSED', 'HALF_OPEN'])}"
    ),
    lambda: (
        f"Retry attempt {random.randint(1, 3)}/{random.randint(3, 5)} for operation={random.choice(['payment.charge', 'notification.send', 'order.create'])}"
    ),
    lambda: (
        f"Message consumed from topic={random.choice(['orders', 'payments', 'notifications', 'analytics'])} partition={random.randint(0, 10)} offset={random.randint(10000, 99999)}"
    ),
    lambda: (
        f"Feature flag evaluated flag={random.choice(['new_checkout', 'dark_mode', 'recommendation_v2', 'live_chat'])} result={random.choice([True, False])} userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Database migration progress version={random.randint(1, 100)} direction={random.choice(['up', 'down'])} table={random.choice(['users', 'orders', 'products', 'reviews'])}"
    ),
    lambda: (
        f"HTTP headers received: x-request-id=req_{random.randint(1000, 9999)}, x-forwarded-for={random.randint(10, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    ),
]

warn_messages = [
    lambda: (
        f"Query execution slower than threshold time={random.randint(500, 5000)}ms threshold=500ms query=SELECT * FROM orders JOIN order_items"
    ),
    lambda: (
        f"Memory usage high heap={random.randint(70, 95)}% used={random.randint(512, 4096)}MB max={random.randint(1024, 8192)}MB"
    ),
    lambda: (
        f"Disk space warning mount={random.choice(['/data', '/var/log', '/app'])} used={random.randint(80, 95)}% available={random.randint(100, 1024)}MB"
    ),
    lambda: (
        f"API response time degraded endpoint={random.choice(['/api/search', '/api/recommend', '/api/export'])} p99={random.randint(2000, 10000)}ms"
    ),
    lambda: (
        f"Database connection pool nearing limit pool={random.choice(['read', 'write'])} used={random.randint(70, 95)}% max={random.randint(50, 200)}"
    ),
    lambda: (
        f"Retry limit approaching operation={random.choice(['payment.charge', 'notification.send', 'order.sync'])} attempts={random.randint(3, 4)} maxAttempts=5"
    ),
    lambda: (
        f"Rate limit threshold crossed userId={random.randint(1000, 99999)} usage={random.randint(80, 99)}% limit={random.randint(100, 1000)}/minute"
    ),
    lambda: (
        f"Certificate expiring soon daysLeft={random.randint(1, 30)} endpoint={random.choice(['api.internal', '*.example.com'])}"
    ),
    lambda: (
        f"Deprecated API version called endpoint=/api/v{random.choice(['1', '2'])}/{random.choice(['users', 'orders'])} suggested=/api/v3/{random.choice(['users', 'orders'])}"
    ),
    lambda: (
        f"Message queue depth growing topic={random.choice(['orders', 'payments', 'notifications'])} depth={random.randint(10000, 100000)} messages"
    ),
    lambda: (
        f"Cache hit rate dropping rate={random.randint(50, 70)}% cache={random.choice(['product_cache', 'user_session', 'price_cache'])}"
    ),
    lambda: (
        f"Thread pool utilization high pool={random.choice(['http', 'async', 'scheduled'])} active={random.randint(80, 100)}% queueDepth={random.randint(100, 1000)}"
    ),
]

TOTAL = 500000
OUTPUT = "D:/alouette-AI/normal_logs_500k.csv"

with open(OUTPUT, "w", encoding="utf-8") as f:
    for i in range(TOTAL):
        level = (
            random.choices(levels, weights=[60, 25, 10, 5])[0]
            if random.random() < 0.97
            else random.choice(levels)
        )
        ts = base_time + timedelta(seconds=random.randint(0, 86399))
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        svc = random.choice(services)

        if level == "INFO":
            msg = random.choice(info_messages)()
        elif level == "DEBUG":
            msg = random.choice(debug_messages)()
        else:  # WARN or TRACE
            msg = random.choice(warn_messages)()

        f.write(f"{level},{ts_str},{svc},{msg}\n")

        if (i + 1) % 50000 == 0:
            print(f"  ... {i + 1}/{TOTAL} lines written")

import os

size = os.path.getsize(OUTPUT)
print(f"\nDone - {TOTAL} normal log lines to {OUTPUT}")
print(f"File size: {size / 1024 / 1024:.1f} MB")

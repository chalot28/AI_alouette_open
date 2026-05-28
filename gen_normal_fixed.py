import csv
import os
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

info_msgs = [
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
    lambda: f"DB query completed in {random.randint(1, 100)}ms",
    lambda: (
        f"Shipment created shipmentId=shp_{random.randint(1000, 9999)} carrier={random.choice(['GHN', 'GHTK', 'VNPost', 'Viettel'])}"
    ),
    lambda: (
        f"Email sent to user={random.randint(1000, 99999)} template={random.choice(['welcome', 'order_confirmation', 'reset_password'])}"
    ),
    lambda: (
        f"API call success endpoint={random.choice(['/api/users', '/api/orders', '/api/products'])} status=200 duration={random.randint(10, 500)}ms"
    ),
    lambda: (
        f"Job completed jobId=job_{random.randint(1000, 9999)} type={random.choice(['cleanup', 'sync', 'backup', 'report'])}"
    ),
    lambda: (
        f"File uploaded userId={random.randint(1000, 99999)} fileSize={random.randint(1024, 1048576)}bytes type={random.choice(['image', 'document', 'video'])}"
    ),
    lambda: (
        f"Inventory updated productId=prod_{random.randint(1000, 9999)} quantity={random.randint(10, 1000)} warehouse=WH-{random.randint(1, 10)}"
    ),
    lambda: (
        f"Webhook delivered hookId=hook_{random.randint(1000, 9999)} event={random.choice(['order.placed', 'payment.received', 'user.registered'])}"
    ),
    lambda: (
        f"Search results query={random.choice(['laptop', 'phone', 'book', 'shoe'])} hits={random.randint(1, 100)} time={random.randint(5, 200)}ms"
    ),
    lambda: (
        f"Token refreshed userId={random.randint(1000, 99999)} expiresIn={random.randint(3600, 86400)}s"
    ),
    lambda: (
        f"Notification pushed deviceId=dev_{random.randint(1000, 9999)} platform={random.choice(['ios', 'android', 'web'])}"
    ),
    lambda: (
        f"Pool healthy pool={random.choice(['read', 'write'])} active={random.randint(1, 30)} idle={random.randint(5, 50)}"
    ),
    lambda: (
        f"Health check passed service={random.choice(['user-service', 'payment-service', 'order-service'])} status=SERVING"
    ),
    lambda: (
        f"Backup completed type={random.choice(['full', 'incremental', 'differential'])} size={random.randint(100, 10000)}MB duration={random.randint(60, 3600)}s"
    ),
]
debug_msgs = [
    lambda: (
        f"Processing {random.choice(['GET', 'POST', 'PUT', 'DELETE'])} /api/v{random.randint(1, 3)}/{random.choice(['users', 'orders', 'products'])}/{random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Deserializing payload orderId=ord_{random.randint(1000, 99999)} format={random.choice(['json', 'protobuf', 'avro'])}"
    ),
    lambda: (
        f"Thread pool active={random.randint(1, 20)} idle={random.randint(5, 30)} queue={random.randint(0, 50)} pool={random.choice(['async', 'io', 'batch'])}"
    ),
    lambda: (
        f"Redis pipeline keys={random.randint(1, 100)} cmds={random.choice(['GET', 'MGET', 'HGETALL', 'LRANGE'])}"
    ),
    lambda: (
        f"SQL params query_{random.randint(100, 999)} userId={random.randint(1000, 99999)} status={random.choice(['ACTIVE', 'PENDING', 'COMPLETED'])}"
    ),
    lambda: (
        f"Cache eviction currentSize={random.randint(1000, 10000)} maxSize={random.randint(5000, 20000)} policy={random.choice(['LRU', 'LFU', 'TTL'])}"
    ),
    lambda: (
        f"JWT decode userId={random.randint(1000, 99999)} algorithm={random.choice(['HS256', 'RS256', 'ES256'])}"
    ),
    lambda: (
        f"Circuit breaker state={random.choice(['CLOSED', 'HALF_OPEN'])} service={random.choice(['payment', 'inventory', 'shipping'])}"
    ),
    lambda: (
        f"Retry {random.randint(1, 3)}/{random.randint(3, 5)} operation={random.choice(['payment.charge', 'notification.send', 'order.create'])}"
    ),
    lambda: (
        f"Consumed topic={random.choice(['orders', 'payments', 'notifications', 'analytics'])} partition={random.randint(0, 10)} offset={random.randint(10000, 99999)}"
    ),
    lambda: (
        f"Feature flag={random.choice(['new_checkout', 'dark_mode', 'recommendation_v2', 'live_chat'])} result={random.choice([True, False])} userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Headers x-request-id=req_{random.randint(1000, 9999)} x-forwarded-for={random.randint(10, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    ),
]
warn_msgs = [
    lambda: f"Query slow time={random.randint(500, 5000)}ms threshold=500ms",
    lambda: (
        f"Memory high heap={random.randint(70, 95)}% used={random.randint(512, 4096)}MB max={random.randint(1024, 8192)}MB"
    ),
    lambda: (
        f"Disk space warning mount={random.choice(['/data', '/var/log', '/app'])} used={random.randint(80, 95)}% available={random.randint(100, 1024)}MB"
    ),
    lambda: (
        f"API degraded endpoint={random.choice(['/api/search', '/api/recommend', '/api/export'])} p99={random.randint(2000, 10000)}ms"
    ),
    lambda: (
        f"DB pool nearing limit pool={random.choice(['read', 'write'])} used={random.randint(70, 95)}% max={random.randint(50, 200)}"
    ),
    lambda: (
        f"Retry limit approaching operation={random.choice(['payment.charge', 'notification.send', 'order.sync'])} attempts={random.randint(3, 4)} max=5"
    ),
    lambda: (
        f"Rate limit threshold userId={random.randint(1000, 99999)} usage={random.randint(80, 99)}% limit={random.randint(100, 1000)}/min"
    ),
    lambda: (
        f"Cert expiring daysLeft={random.randint(1, 30)} endpoint={random.choice(['api.internal', '*.example.com'])}"
    ),
    lambda: (
        f"Deprecated API /api/v{random.choice(['1', '2'])}/{random.choice(['users', 'orders'])} use /api/v3"
    ),
    lambda: (
        f"Queue depth growing topic={random.choice(['orders', 'payments', 'notifications'])} depth={random.randint(10000, 100000)}"
    ),
    lambda: (
        f"Cache hit rate dropping rate={random.randint(50, 70)}% cache={random.choice(['product_cache', 'user_session', 'price_cache'])}"
    ),
    lambda: (
        f"Thread pool high pool={random.choice(['http', 'async', 'scheduled'])} active={random.randint(80, 100)}% queue={random.randint(100, 1000)}"
    ),
]

CSV_PATH = "D:/alouette-AI/data/raw/normal_logs_500k.csv"
TOTAL = 500000

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    for i in range(TOTAL):
        level = (
            random.choices(levels, weights=[60, 25, 10, 5])[0]
            if random.random() < 0.97
            else random.choice(levels)
        )
        ts = (base_time + timedelta(seconds=random.randint(0, 86399))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        svc = random.choice(services)
        if level == "INFO":
            msg = random.choice(info_msgs)()
        elif level == "DEBUG":
            msg = random.choice(debug_msgs)()
        else:
            msg = random.choice(warn_msgs)()
        # Normal logs chỉ có 4 cột (không có exception)
        writer.writerow([level, ts, svc, msg])
        if (i + 1) % 50000 == 0:
            print(f"  ... {i + 1}/{TOTAL}")

size = os.path.getsize(CSV_PATH)
print(f"Done - {TOTAL} lines, {size / 1024 / 1024:.1f} MB")

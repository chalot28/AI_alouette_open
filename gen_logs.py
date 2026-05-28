import random
from datetime import datetime, timedelta

random.seed(42)

levels = ["ERROR", "FATAL"]

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

exceptions = [
    "NullPointerException",
    "SQLException",
    "TimeoutException",
    "IllegalArgumentException",
    "ConnectionRefusedException",
    "OutOfMemoryError",
    "IndexOutOfBoundsException",
    "HttpClientErrorException",
    "ArithmeticException",
    "ClassCastException",
    "ConcurrentModificationException",
    "CustomAuthenticationException",
    "RateLimitExceededException",
    "SerializationException",
    "ValidationException",
    "ResourceNotFoundException",
    "IllegalStateException",
    "UnsupportedOperationException",
    "SecurityException",
    "IOException",
]

base_time = datetime(2026, 5, 28, 0, 0, 0)

msgs = {}

msgs["NullPointerException"] = [
    lambda: (
        f"Cannot invoke getId() because user is null userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Cannot read field email because result is null requestId=req-{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Cannot invoke equals() because key is null cacheKey={random.randint(10000, 99999)}"
    ),
    lambda: (
        f"Session object is null when processing login sessionId=sess_{random.randint(1000, 9999)}"
    ),
    lambda: f"Profile data is null for userId={random.randint(1000, 99999)}",
    lambda: (
        f"Cannot invoke toString() on null object at OrderProcessor.process() orderId=ord_{random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Address is null for shipping label userId={random.randint(1000, 99999)} addressType={random.choice(['billing', 'shipping'])}"
    ),
    lambda: (
        f"Configuration value not set for key={random.choice(['db.url', 'redis.host', 'kafka.bootstrap'])}"
    ),
    lambda: (
        f"Callback handler is null websocketSession=ws_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Parent entity is null cannot persist child entity={random.choice(['OrderItem', 'Invoice', 'Shipment'])}"
    ),
    lambda: (
        f"Database connection metadata is null host={random.choice(['primary', 'replica-1', 'replica-2'])}"
    ),
    lambda: (
        f"Response body is null after HTTP call to {random.choice(['/api/users', '/api/orders', '/api/payments'])}"
    ),
    lambda: (
        f"Certificate chain is null for SSL handshake endpoint={random.choice(['api.internal', 'gateway.public'])}"
    ),
    lambda: (
        f"Event payload is null in consumer group={random.choice(['order-group', 'payment-group', 'notification-group'])}"
    ),
    lambda: (
        f"ThreadLocal variable not initialized for requestId={random.randint(1000, 9999)} userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Resolver returned null for dependency type={random.choice(['Repository', 'Service', 'Client'])} name={random.choice(['UserRepo', 'PaymentService', 'RedisClient'])}"
    ),
]

msgs["SQLException"] = [
    lambda: (
        f"Connection pool exhausted after {random.randint(10, 60)}s activeConnections={random.randint(50, 200)}"
    ),
    lambda: (
        f"Deadlock detected when updating order orderId=ord_{random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Duplicate key violation on users.email email=user{random.randint(1, 9999)}@test.com"
    ),
    lambda: (
        f"Query timeout exceeded {random.randint(30, 300)}s query=SELECT * FROM orders WHERE userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Cannot insert null into column status table=transactions txId=tx_{random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Table orders_2025_01 not found schema={random.choice(['public', 'audit', 'archive'])}"
    ),
    lambda: (
        f"Disk full while writing WAL segment database={random.choice(['postgresql', 'mysql', 'mariadb'])} host=db-{random.randint(1, 5)}"
    ),
    lambda: (
        f"Maximum prepared transactions reached limit={random.randint(100, 1000)} current={random.randint(100, 1000)}"
    ),
    lambda: (
        f"Foreign key constraint fails on order_items.order_id value=ord_{random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Serialization failure on transaction txId=tx_{random.randint(1000, 99999)} retry={random.randint(0, 3)}"
    ),
    lambda: (
        f"Column account_balance does not exist in table users schema={random.choice(['v1', 'v2', 'migrating'])}"
    ),
    lambda: (
        f"Cannot truncate table audit_log because of pending triggers triggerName={random.choice(['log_insert', 'log_update', 'log_delete'])}"
    ),
    lambda: (
        f"Database connection string malformed expected format host:port got={random.choice(['localhost', '127.0.0.1', ':3306'])}"
    ),
    lambda: (
        f"Transaction log seq mismatch expected={random.randint(1000, 9999)} actual={random.randint(1000, 9999)} database={random.choice(['mysql-bin', 'pg_wal'])}"
    ),
    lambda: (
        f"Cannot acquire advisory lock key={random.randint(10000, 99999)} session={random.randint(100, 999)}"
    ),
    lambda: (
        f"Function get_user_orders(INT) does not exist schema={random.choice(['public', 'app'])}"
    ),
    lambda: f"Date overflow value 2026-13-01 for column expiry_date table=coupons",
]

msgs["TimeoutException"] = [
    lambda: (
        f"Payment gateway did not respond within {random.randint(5, 60)}s orderId=ord_{random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Redis GET timed out after {random.randint(2, 30)}s key=session:{random.randint(1000, 99999)}"
    ),
    lambda: (
        f"External API call to /v2/verify timed out after {random.randint(3, 30)}s retryAttempt={random.randint(0, 3)}"
    ),
    lambda: f"Database health check timeout ms={random.randint(5000, 30000)}",
    lambda: (
        f"Kafka produce message timed out topic=order-events partition={random.randint(0, 10)}"
    ),
    lambda: (
        f"gRPC call to rating-service.BatchGet timed out after {random.randint(5, 60)}s deadline={random.randint(5, 60)}s"
    ),
    lambda: (
        f"Elasticsearch search query timed out after {random.randint(10, 120)}s index=orders indexMode={random.choice(['search', 'analytics'])}"
    ),
    lambda: (
        f"File upload stream idle timeout exceeded {random.randint(30, 300)}s uploadId=up_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Consul service discovery timed out for service={random.choice(['payment', 'inventory', 'shipping'])} datacenter=dc-{random.randint(1, 3)}"
    ),
    lambda: (
        f"Aggregation pipeline timed out after {random.randint(30, 300)}s pipeline={random.choice(['/reports/daily', '/analytics/revenue', '/metrics/mau'])}"
    ),
    lambda: (
        f"Distributed lock acquire timed out lockKey=order:{random.randint(1000, 99999)} currentHolder=pod-{random.randint(1, 10)}"
    ),
    lambda: (
        f"WebSocket ping-pong timeout sessionId=ws_{random.randint(1000, 9999)} lastPong={random.randint(30, 120)}s ago"
    ),
    lambda: (
        f"S3 presigned URL generation timed out bucket={random.choice(['uploads', 'exports', 'backups'])} region={random.choice(['us-east-1', 'ap-southeast-1', 'eu-west-1'])}"
    ),
    lambda: (
        f"Hazelcast map operation timed out cluster=hazelcast-{random.choice(['prod', 'dr', 'staging'])} operation=map.get"
    ),
    lambda: (
        f"LDAP authentication timeout ms={random.randint(5000, 30000)} server=ldap-{random.randint(1, 3)}.internal"
    ),
    lambda: (
        f"Thread pool queue full for executor={random.choice(['async-job', 'payment-worker', 'notification-push'])} queueSize={random.randint(1000, 10000)}"
    ),
    lambda: (
        f"AJAX long-poll request timed out for client={random.randint(1000, 9999)} timeoutMs={random.randint(30000, 120000)}"
    ),
]

msgs["IllegalArgumentException"] = [
    lambda: (
        f"Negative amount not allowed amount={random.randint(-10000, -1)} userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Invalid email format provided email={random.choice(['abc@', 'test@@', '@.com', 'user', '.@mail.com', 'a@b@c.com'])}"
    ),
    lambda: (
        f"Page index must be >= 0 requested={random.randint(-10, -1)} pageSize={random.randint(10, 100)}"
    ),
    lambda: (
        f"Order status invalid for cancel currentStatus={random.choice(['SHIPPED', 'DELIVERED', 'RETURNED'])} orderId=ord_{random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Currency code not supported currency={random.choice(['XYZ', 'ABC', 'BTC', 'GOLD', 'INR', 'TEST'])} amount={random.randint(1, 1000)}"
    ),
    lambda: (
        f"Password must be at least 8 characters received={random.randint(1, 7)} userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Invalid UUID format input={random.choice(['abc', '123', 'xyz-abc', 'not-uuid'])} paramName={random.choice(['orderId', 'transactionId', 'sessionId'])}"
    ),
    lambda: (
        f"Discount percentage out of range value={random.randint(-50, 150)}% maxAllowed=50"
    ),
    lambda: (
        f"SKU format invalid sku={random.choice(['ABC', '12345', '!@#', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'])}"
    ),
    lambda: (
        f"Birth date cannot be in the future date={random.choice(['2099-01-01', '2100-12-31', '3000-01-01'])} userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Phone number must be numeric received={random.choice(['+84 (0) 123 456', 'abc123', '(84) 9123'])}"
    ),
    lambda: (
        f"Sort field not whitelisted field={random.choice(['DROP TABLE', 'undefined', 'constructor', '__proto__'])}"
    ),
    lambda: (
        f"JSON path expression invalid path={random.choice(['$.', '..', '[0]', '$[abc]'])} documentId=doc_{random.randint(100, 999)}"
    ),
    lambda: (
        f"Time range start must be before end start={random.choice(['2026-05-29', '2026-06-01', '2026-12-31'])} end=2026-05-28"
    ),
    lambda: (
        f"File extension not allowed extension={random.choice(['.exe', '.bat', '.sh', '.php', '.js'])} maxSize={random.randint(1, 10)}MB"
    ),
    lambda: (
        f"Rating value out of bounds value={random.randint(-5, 10)} allowedRange=1-5 productId=prod_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Coupon code has invalid checksum code={random.choice(['SAVE50', 'GET10', 'FLASH25', 'WELCOME'])} length={random.randint(1, 20)}"
    ),
]

msgs["ConnectionRefusedException"] = [
    lambda: (
        f"Cannot connect to MySQL at host=db-primary-{random.randint(1, 3)}.internal port=3306 error=econnrefused"
    ),
    lambda: (
        f"Redis master at 10.0.{random.randint(10, 99)}.{random.randint(1, 254)}:6379 refused connection retry={random.randint(0, 5)}"
    ),
    lambda: (
        f"RabbitMQ broker unavailable host=rabbit-{random.randint(1, 5)}.cluster.local vhost=/production"
    ),
    lambda: (
        f"Elasticsearch node not reachable endpoint=https://es-{random.choice(['data', 'master'])}-{random.randint(1, 5)}:9200"
    ),
    lambda: (
        f"gRPC connection refused target=inventory-service:{random.randint(5000, 5100)} errorCode=ECONNREFUSED"
    ),
    lambda: (
        f"MongoDB replica set primary unreachable host=mongo-{random.choice(['0', '1', '2'])}.node.local port=27017"
    ),
    lambda: (
        f"SSH connection refused to deploy@10.0.{random.randint(1, 10)}.{random.randint(1, 254)}:{random.randint(22, 2222)}"
    ),
    lambda: (
        f"Socket connection refused when connecting to smtp-{random.randint(1, 3)}.internal:{random.randint(25, 587)} timeout={random.randint(5, 30)}s"
    ),
    lambda: (
        f"Cassandra node down host=cassandra-{random.randint(1, 6)}.datastax.internal port=9042 keyspace={random.choice(['orders', 'users', 'inventory'])}"
    ),
    lambda: (
        f"HTTP proxy connection refused proxy=proxy-{random.randint(1, 5)}.corp.net:{random.randint(3128, 8080)} target={random.choice(['api.external.com', 'cdn.internal'])}"
    ),
    lambda: (
        f"Kafka broker {random.choice(['kafka-0', 'kafka-1', 'kafka-2'])}.kafka.cluster.local:9093 refused connection controllerId={random.randint(1, 3)}"
    ),
    lambda: (
        f"NFS mount failed server=nfs-{random.randint(1, 3)}.storage.internal:{random.choice(['/exports/data', '/exports/logs', '/exports/backup'])}"
    ),
    lambda: (
        f"Docker daemon socket refused at tcp://{random.randint(10, 99)}.{random.randint(10, 99)}.{random.randint(1, 10)}.{random.randint(1, 254)}:2375"
    ),
    lambda: (
        f"Thrift connection refused to service={random.choice(['profile-service', 'feed-service', 'ads-service'])} port={random.randint(9000, 9999)}"
    ),
    lambda: (
        f"Memcached connection refused at {random.choice(['memcached-1', 'memcached-2', 'memcached-3'])}.cache.internal:11211"
    ),
    lambda: (
        f"WebSocket handshake failed endpoint=wss://socket-{random.randint(1, 5)}.app.io/ws origin=https://app.io"
    ),
    lambda: (
        f"FTP connection refused to ftp-{random.choice(['uploads', 'exports', 'mirror'])}.internal:{random.randint(21, 2121)}"
    ),
]

msgs["OutOfMemoryError"] = [
    lambda: (
        f"Java heap space 2GB exhausted during large CSV export requestedRows={random.randint(500000, 5000000)}"
    ),
    lambda: (
        f"Metaspace overflow after loading {random.randint(10000, 50000)} classes leak app=recommendation-engine"
    ),
    lambda: (
        f"Unable to allocate {random.randint(100, 1024)}MB buffer for image processing fileSize={random.randint(500, 2000)}MB"
    ),
    lambda: (
        f"Direct memory limit exceeded allocated={random.randint(1024, 8192)}MB max={random.randint(512, 4096)}MB"
    ),
    lambda: (
        f"GC overhead limit exceeded 98pct time spent in GC with {random.randint(50, 200)} threads active"
    ),
    lambda: (
        f"Native memory allocation failed for thread stack size={random.randint(1024, 4096)}KB threadCount={random.randint(500, 5000)}"
    ),
    lambda: (
        f"Compressed class space used={random.randint(512, 2048)}MB max={random.randint(256, 1024)}MB"
    ),
    lambda: f"Could not reserve enough space for {random.randint(2, 16)}GB object heap",
    lambda: (
        f"String table overflow size={random.randint(500000, 5000000)} entries limit={random.randint(100000, 1000000)}"
    ),
    lambda: (
        f"Code cache exhausted used={random.randint(200, 500)}MB max={random.randint(128, 256)}MB JIT compilation stopped"
    ),
    lambda: (
        f"GPU out of memory when processing batch batchSize={random.randint(64, 512)} tensorShape={random.choice(['[512,768]', '[1024,512]', '[2048,256]'])}"
    ),
    lambda: (
        f"StackOverflowError when resolving recursive dependency chain depth={random.randint(1000, 10000)}"
    ),
    lambda: (
        f"Unable to create new native thread for request requestId=req-{random.randint(1000, 9999)} limit={random.randint(1000, 10000)}"
    ),
    lambda: (
        f"Redis OOM command not allowed when used memory > maxmemory used={random.randint(1, 8)}GB max=8GB"
    ),
    lambda: (
        f"Node.js heap out of memory allocationFailed size={random.randint(500, 4000)}MB"
    ),
    lambda: (
        f"Python MemoryError allocating numpy array shape=({random.randint(10000, 100000)},{random.randint(1000, 10000)}) dtype=float64"
    ),
    lambda: (
        f"MapReduce task memory exceeded container memory={random.randint(4, 32)}GB requested={random.randint(8, 64)}GB taskAttempt={random.randint(1, 5)}"
    ),
]

msgs["IndexOutOfBoundsException"] = [
    lambda: (
        f"Array index {random.randint(100, 999)} out of bounds for length {random.randint(10, 99)} in batch batchId=batch_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Index -1 accessed in order line items orderId=ord_{random.randint(1000, 99999)} itemsCount={random.randint(0, 5)}"
    ),
    lambda: (
        f"String index out of range {random.randint(50, 200)} when parsing webhook payloadId=hook_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"List index {random.randint(100, 500)} out of bounds for size {random.randint(1, 99)} in pagination page={random.randint(10, 50)}"
    ),
    lambda: (
        f"Buffer underflow reading packet at offset {random.randint(-100, -1)} protocolVersion={random.randint(1, 5)}"
    ),
    lambda: (
        f"Row index {random.randint(1000, 9999)} out of bounds for ResultSet with {random.randint(10, 999)} rows queryId=qry_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Character position {random.randint(500, 5000)} out of range for text length {random.randint(10, 499)} in parser"
    ),
    lambda: (
        f"Byte array index {random.randint(1000, 9999)} out of bounds for length {random.randint(10, 999)} when decrypting payload"
    ),
    lambda: (
        f"Stack index {random.randint(100, 500)} out of bounds depth={random.randint(10, 99)} in expression evaluator"
    ),
    lambda: (
        f"Column index {random.randint(50, 200)} out of range schema has {random.randint(1, 49)} columns table={random.choice(['users', 'orders', 'products'])}"
    ),
    lambda: (
        f"Vertex index {random.randint(1000, 9999)} out of bounds graph has {random.randint(100, 999)} nodes"
    ),
    lambda: (
        f"Frame buffer line {random.randint(1000, 9999)} out of range height={random.randint(100, 999)}px videoFrameId=vf_{random.randint(100, 999)}"
    ),
    lambda: (
        f"Time series data point {random.randint(10000, 99999)} exceeds array capacity {random.randint(1000, 9999)} metric={random.choice(['cpu', 'memory', 'latency', 'throughput'])}"
    ),
    lambda: (
        f"Chunk index {random.randint(100, 999)} out of bounds totalChunks={random.randint(1, 99)} uploadId=up_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Grid cell index ({random.randint(0, 9)},{random.randint(10, 99)}) out of bounds gridSize={random.randint(1, 9)}"
    ),
    lambda: (
        f"Embedding dimension {random.randint(1000, 9999)} out of range modelDim={random.randint(128, 768)}"
    ),
]

msgs["HttpClientErrorException"] = [
    lambda: (
        f"404 Not Found from /api/v3/users/{random.randint(1000, 99999)} upstream=user-service responseTime={random.randint(100, 5000)}ms"
    ),
    lambda: (
        f"403 Forbidden on resource /admin/orders/{random.randint(1000, 99999)} missingScope=admin:write"
    ),
    lambda: (
        f"429 Too Many Requests from rate-limiter retryAfter={random.randint(1, 120)}s limit={random.randint(10, 1000)}/minute"
    ),
    lambda: (
        f"502 Bad Gateway from payment-service/authorize upstreamTimeout={random.randint(10, 60)}s"
    ),
    lambda: (
        f"503 Service Unavailable inventory-api circuitBreaker=OPEN fallbackFailed=true"
    ),
    lambda: (
        f'400 Bad Request from /api/v2/checkout body={{"errors":["invalid_payload","missing_field"]}} referenceId=ref_{random.randint(1000, 9999)}'
    ),
    lambda: (
        f"401 Unauthorized accessing /api/internal/reports token={random.choice(['expired', 'invalid', 'revoked'])} userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"405 Method Not Allowed {random.choice(['PATCH', 'DELETE', 'PUT'])} /api/v1/orders/{random.randint(1000, 99999)} allow=GET,POST"
    ),
    lambda: (
        f"408 Request Timeout from search-service/_search time={random.randint(30, 120)}s cluster={random.choice(['green', 'yellow', 'red'])}"
    ),
    lambda: (
        f"413 Payload Too Large uploadSize={random.randint(10, 100)}MB maxAllowed={random.randint(1, 10)}MB endpoint=/api/v3/files/upload"
    ),
    lambda: (
        f"422 Unprocessable Entity from /api/v2/payments validationErrors={random.randint(1, 5)} txId=tx_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"500 Internal Server Error from recommendation-service/rank upstream=ml-engine traceId=trace_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"504 Gateway Timeout after {random.randint(30, 120)}s calling shipping-service/track upstreamTimeout={random.randint(30, 120)}s"
    ),
    lambda: (
        f"301 Redirect loop detected redirectCount={random.randint(10, 50)} url={random.choice(['/old/api', '/v1/legacy', '/deprecated'])}"
    ),
    lambda: (
        f"415 Unsupported Media Type contentType={random.choice(['text/plain', 'application/xml', 'multipart/form-data'])} endpoint=/api/v2/graphql"
    ),
    lambda: (
        f"507 Insufficient Storage on upload-service remaining={random.randint(0, 100)}MB quota={random.randint(1024, 10240)}MB userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"431 Request Header Fields Too Large headerSize={random.randint(10, 100)}KB maxAllowed=8KB requestId=req_{random.randint(1000, 9999)}"
    ),
]

msgs["ArithmeticException"] = [
    lambda: (
        f"Division by zero in rating calculation avgRating=NaN productId=prod_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Integer overflow when calculating price total={random.randint(2147483000, 2147483647)} * {random.randint(2, 100)}"
    ),
    lambda: (
        f"Floating point underflow in loss computation value={random.choice(['Infinity', '-Infinity', 'NaN'])} epoch={random.randint(1, 500)}"
    ),
    lambda: (
        f"Rounding mode needed for BigDecimal division value=1/3 scale={random.randint(10, 50)}"
    ),
    lambda: (
        f"Modulo by zero in partition calculation partitionKey={random.randint(1, 100)} totalPartitions=0"
    ),
    lambda: (
        f"Exact arithmetic overflow when summing {random.randint(1000000, 9999999)} items total={random.randint(922337203685477, 9223372036854775)}"
    ),
    lambda: (
        f"Fractional conversion error for currency value={random.randint(1, 9999)}.{random.randint(0, 9999)} currency={random.choice(['VND', 'IDR', 'JPY', 'KRW'])}"
    ),
]

msgs["ClassCastException"] = [
    lambda: (
        f"Cannot cast org.json.JSONObject to java.util.List field=data payloadType={random.choice(['json', 'xml', 'protobuf'])}"
    ),
    lambda: (
        f"Cannot cast Proxy$107 to com.app.service.UserService dynamicProxy JDK version={random.choice(['8', '11', '17', '21'])}"
    ),
    lambda: (
        f"Cannot cast String to Integer when parsing config key={random.choice(['max.retries', 'pool.size', 'timeout.ms'])} value={random.choice(['ten', 'unlimited', 'auto'])}"
    ),
    lambda: f"Cannot cast ResponseEntity to OrderResponse in controller advice",
    lambda: (
        f"Cannot cast SessionImpl to HttpSession in filter chain module={random.choice(['security', 'audit', 'cors'])}"
    ),
    lambda: (
        f"Cannot cast LinkedHashMap to OrderDTO when deserializing redis cache key=order:{random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Cannot cast Mono to Flux in reactive pipeline operator={random.choice(['flatMap', 'concatMap', 'switchIfEmpty'])}"
    ),
]

msgs["ConcurrentModificationException"] = [
    lambda: (
        f"Order list modified while iterating over shipment items orderId=ord_{random.randint(1000, 99999)} thread={random.choice(['pool-1', 'pool-2', 'main'])}"
    ),
    lambda: (
        f"Session cache modified during flush userId={random.randint(1000, 99999)} sessionId=sess_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Concurrent map update detected on inventory stock productId=prod_{random.randint(1000, 9999)} warehouse=WH-{random.randint(1, 10)}"
    ),
    lambda: (
        f"Event list modified while publishing batch batchSize={random.randint(10, 100)} topic={random.choice(['order.events', 'payment.events', 'user.events'])}"
    ),
    lambda: (
        f"Configuration map mutated during hot reload key={random.choice(['feature.flags', 'rate.limits', 'endpoints'])}"
    ),
    lambda: (
        f"Cache entry expired while iterating cache={random.choice(['session', 'user', 'product', 'price'])} size={random.randint(10000, 100000)}"
    ),
]

msgs["CustomAuthenticationException"] = [
    lambda: (
        f"JWT token expired at {random.choice(['2026-05-27', '2026-05-26', '2026-05-25'])} userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Invalid API key provided key={random.choice(['sk_live_xxxx', 'sk_test_xxxx', 'pk_xxxx'])[: random.randint(10, 20)]}... origin={random.choice(['example.com', 'unknown.org', '192.168.1.1'])}"
    ),
    lambda: (
        f"OTP verification failed attempts={random.randint(1, 5)} phone=+84{random.randint(90000000, 99999999)}"
    ),
    lambda: (
        f"Refresh token revoked or not found tokenId=tok_{random.randint(1000, 9999)} userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Two-factor authentication code expired code={random.randint(100000, 999999)} issuedAt={random.randint(30, 300)}s ago"
    ),
    lambda: (
        f"Session hijacking detected IP mismatch sessionIp=10.0.{random.randint(1, 254)}.{random.randint(1, 254)} requestIp={random.randint(1, 223)}.{random.randint(1, 223)}.{random.randint(1, 223)}.{random.randint(1, 223)}"
    ),
    lambda: (
        f"SAML assertion invalid signature issuer={random.choice(['idp.example.com', 'adfs.corp.net', 'okta.internal'])}"
    ),
    lambda: (
        f"OAuth2 authorization code already used code=auth_{random.randint(1000, 9999)} clientId={random.randint(10000, 99999)}"
    ),
    lambda: (
        f"Password hash mismatch for userId={random.randint(1000, 99999)} algorithm={random.choice(['bcrypt', 'scrypt', 'argon2', 'pbkdf2'])}"
    ),
    lambda: (
        f"Certificate CN mismatch expected=api.{random.choice(['prod', 'staging', 'dev'])}.example.com got={random.choice(['*.example.com', 'localhost', 'other.com'])}"
    ),
    lambda: (
        f"Rate limit exceeded for login endpoint userId={random.randint(1000, 99999)} window={random.randint(1, 15)}min count={random.randint(10, 100)}"
    ),
    lambda: (
        f"Device fingerprint mismatch userId={random.randint(1000, 99999)} deviceId={random.choice(['unknown', 'changed', 'spoofed'])}"
    ),
]

msgs["RateLimitExceededException"] = [
    lambda: (
        f"API rate limit exceeded for key={random.choice(['sk_live', 'pk_live', 'sk_test'])}_{random.randint(1000, 9999)} limit={random.randint(100, 10000)}/hour"
    ),
    lambda: (
        f"Too many concurrent requests from IP {random.randint(1, 223)}.{random.randint(1, 223)}.{random.randint(1, 223)}.{random.randint(1, 254)} limit={random.randint(10, 100)}"
    ),
    lambda: (
        f"Endpoint /api/v3/search rate limited burst={random.randint(10, 100)}/s sustained={random.randint(50, 500)}/minute"
    ),
    lambda: (
        f"User {random.randint(1000, 99999)} exceeded subscription quota used={random.randint(100, 1000)} limit={random.randint(100, 1000)} plan={random.choice(['free', 'basic', 'premium', 'enterprise'])}"
    ),
    lambda: (
        f"Database connection rate limit hit pool={random.choice(['read', 'write', 'reporting'])} active={random.randint(50, 200)} max={random.randint(50, 200)}"
    ),
    lambda: (
        f"SMS OTP rate limit exceeded phone=+84{random.randint(90000000, 99999999)} sent={random.randint(3, 10)} window=5min"
    ),
    lambda: (
        f"Export API rate limited for userId={random.randint(1000, 99999)} remaining={random.randint(0, 5)} resetsIn={random.randint(60, 3600)}s"
    ),
    lambda: (
        f"File upload rate limit exceeded for IP {random.randint(1, 223)}.{random.randint(1, 223)}.{random.randint(1, 223)}.{random.randint(1, 254)} limit={random.randint(5, 30)}/minute"
    ),
]

msgs["SerializationException"] = [
    lambda: (
        f"Cannot deserialize class com.app.model.Order from JSON version mismatch expected=2 got={random.randint(3, 10)}"
    ),
    lambda: (
        f"Protobuf message size exceeded limit={random.randint(4, 64)}MB messageSize={random.randint(8, 128)}MB type={random.choice(['OrderEvent', 'PaymentEvent'])}"
    ),
    lambda: (
        f"Kafka message deserialization failed for topic={random.choice(['orders', 'payments', 'notifications'])} partition={random.randint(0, 10)} offset={random.randint(10000, 99999)}"
    ),
    lambda: (
        f"Avro schema backward incompatible reader={random.randint(1, 10)} writer={random.randint(11, 20)} recordType={random.choice(['UserV1', 'UserV2', 'UserV3'])}"
    ),
    lambda: (
        f"XML parsing error at line {random.randint(10, 1000)} column {random.randint(1, 100)} documentId=doc_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Java serialization UID mismatch local={random.randint(100000, 999999)} remote={random.randint(100000, 999999)} class=com.app.model.Order"
    ),
    lambda: (
        f"MessagePack unpacking error type={random.choice(['bin', 'str', 'array', 'map', 'ext'])} byteOffset={random.randint(0, 1000)}"
    ),
    lambda: (
        f"YAML parsing error while loading config file={random.choice(['application.yml', 'bootstrap.yml', 'logback.xml'])} key={random.choice(['spring.datasource', 'server.port', 'logging.level'])}"
    ),
    lambda: (
        f"Redis value cannot be deserialized key=session:{random.randint(1000, 99999)} expected=SessionData got=String"
    ),
    lambda: (
        f"BSON document too large max={random.randint(4, 32)}MB size={random.randint(8, 64)}MB collection={random.choice(['audit.logs', 'analytics.events', 'user.sessions'])}"
    ),
]

msgs["ValidationException"] = [
    lambda: (
        f"Order total {random.randint(100000, 999999)} exceeds maximum {random.randint(10000, 99999)} orderId=ord_{random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Missing required field={random.choice(['email', 'phone', 'address', 'fullName'])} in user registration requestId=req_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Credit card number failed Luhn check card=xxxx-xxxx-xxxx-{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Invalid date range start={random.choice(['2026-05-29', '2026-06-01', '2026-07-01'])} end=2026-05-28 bookingId=bk_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Stock insufficient requested={random.randint(10, 100)} available={random.randint(0, 9)} productId=prod_{random.randint(1000, 9999)} warehouse=WH-{random.randint(1, 10)}"
    ),
    lambda: (
        f"Password must contain uppercase, lowercase, digit, special char userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Age must be between 18 and 120 provided={random.randint(0, 17)} userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"URL format invalid url={random.choice(['ftp://bad', 'http://', 'javascript:alert(1)', 'data:text/html'])}"
    ),
    lambda: (
        f"IP address not in whitelist ip={random.randint(1, 223)}.{random.randint(1, 223)}.{random.randint(1, 223)}.{random.randint(1, 254)}"
    ),
    lambda: (
        f"Reference {random.choice(['orderId', 'paymentId', 'shipmentId'])}={random.choice(['null', '', 'undefined', 'nil'])} is required"
    ),
]

msgs["ResourceNotFoundException"] = [
    lambda: f"User not found userId={random.randint(1000, 99999)}",
    lambda: (
        f"Order not found orderId=ord_{random.randint(1000, 99999)} userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Product not found productId=prod_{random.randint(1000, 9999)} sku={random.choice(['SKU001', 'SKU002', 'SKU003'])}"
    ),
    lambda: (
        f"File not found path={random.choice(['/data/exports', '/uploads/images', '/backups/daily'])}/file_{random.randint(1000, 9999)}.csv"
    ),
    lambda: (
        f"Configuration key={random.choice(['feature.toggle.x', 'integration.webhook.url', 'payment.processor'])} not found in namespace={random.choice(['app', 'third-party', 'experiment'])}"
    ),
    lambda: (
        f"Shipment tracking not found trackingId=TRACK{random.randint(100000, 999999)} carrier={random.choice(['GHN', 'GHTK', 'VNPost', 'Viettel'])}"
    ),
    lambda: (
        f"Coupon code={random.choice(['SAVE20', 'WELCOME10', 'FREESHIP', 'FLASH50'])} not found or expired"
    ),
    lambda: (
        f"API endpoint not found method={random.choice(['GET', 'POST', 'PUT', 'DELETE'])} path=/api/v{random.randint(1, 5)}/{random.choice(['users', 'orders', 'payments', 'products'])}/{random.choice(['undefined', 'null', 'unknown'])}"
    ),
    lambda: (
        f"Ticket not found ticketId=TKT-{random.randint(10000, 99999)} userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Cache key not found key={random.choice(['user_profile', 'order_summary', 'product_detail'])}:{random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Bucket not found name={random.choice(['data-lake', 'logs-archive', 'ml-models', 'static-assets'])}-{random.choice(['prod', 'staging', 'dev'])}"
    ),
    lambda: (
        f"Audit trail not found for transaction txId=tx_{random.randint(1000, 99999)} date=2026-05-28"
    ),
    lambda: (
        f"Notification template not found templateId={random.choice(['order_confirmation', 'welcome_email', 'reset_password', 'otp_sms'])} locale={random.choice(['vi_VN', 'en_US', 'ja_JP', 'fr_FR'])}"
    ),
]

msgs["IllegalStateException"] = [
    lambda: (
        f"Cannot process payment when order is in {random.choice(['DRAFT', 'CANCELLED', 'REFUNDED'])} status orderId=ord_{random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Service already initialized cannot reconfigure service={random.choice(['RedisConnector', 'KafkaProducer', 'ElasticClient'])}"
    ),
    lambda: (
        f"Connection pool already closed connections={random.randint(0, 50)} inPool={random.randint(0, 50)}"
    ),
    lambda: (
        f"Transaction already completed txId=tx_{random.randint(1000, 99999)} status={random.choice(['COMMITTED', 'ROLLED_BACK', 'UNKNOWN'])}"
    ),
    lambda: (
        f"Cannot cancel shipment when status={random.choice(['IN_TRANSIT', 'DELIVERED', 'RETURNED'])} shipmentId=shp_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Scheduler already running jobId=job_{random.randint(1000, 9999)} trigger={random.choice(['cron', 'fixedDelay', 'interval'])}"
    ),
    lambda: (
        f"Cannot write to closed output stream bufferSize={random.randint(1024, 65536)} remaining={random.randint(0, 1024)}"
    ),
    lambda: (
        f"File handle already released descriptor={random.randint(0, 9999)} path={random.choice(['/tmp/upload', '/var/log/app', '/data/temp'])}"
    ),
]

msgs["UnsupportedOperationException"] = [
    lambda: (
        f"Mutation not allowed on immutable object type={random.choice(['OrderRecord', 'UserSnapshot', 'ConfigEntry'])}"
    ),
    lambda: (
        f"Operation not supported for database type={random.choice(['H2', 'SQLite', 'HSQLDB'])} required={random.choice(['PostgreSQL', 'MySQL', 'Oracle'])}"
    ),
    lambda: (
        f"Feature not implemented method={random.choice(['batchDelete', 'bulkUpdate', 'streamExport'])} in version={random.choice(['1.0', '1.5', '2.0-beta'])}"
    ),
    lambda: (
        f"Sort not supported for field type={random.choice(['JSON', 'ARRAY', 'GEOMETRY'])} field={random.choice(['metadata', 'tags', 'coordinates'])}"
    ),
    lambda: (
        f"Encryption algorithm not available in current JDK algorithm={random.choice(['AES-256-GCM', 'RSA-OAEP', 'ECIES'])} JDK={random.choice(['8', '11', '17'])}"
    ),
    lambda: (
        f"Write operation not allowed on replica node={random.choice(['replica-1', 'replica-2', 'read-only-follower'])}"
    ),
    lambda: (
        f"Cross-region query not supported regionFrom={random.choice(['us-east-1', 'ap-southeast-1', 'eu-west-1'])} regionTo={random.choice(['us-east-1', 'ap-southeast-1', 'eu-west-1'])}"
    ),
    lambda: (
        f"Cassandra secondary index not supported on column={random.choice(['description', 'metadata', 'payload'])} type={random.choice(['text', 'blob', 'frozen'])}"
    ),
]

msgs["SecurityException"] = [
    lambda: (
        f"Access denied for userId={random.randint(1000, 99999)} resource=/admin/{random.choice(['users', 'orders', 'config', 'logs'])}/{random.randint(1000, 99999)} role={random.choice(['USER', 'GUEST', 'ANONYMOUS'])}"
    ),
    lambda: (
        f"SQL injection attempt detected input={random.choice(['1; DROP TABLE users', "' OR '1'='1", "admin'--", '1 UNION SELECT * FROM passwords'])} userId={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"XSS payload detected in field={random.choice(['fullName', 'description', 'comment'])} content={random.choice(['<script>alert(1)</script>', '<img onerror=alert(1)>', 'javascript:alert(1)'])}"
    ),
    lambda: (
        f"CSRF token mismatch expected=csrf_{random.randint(1000, 9999)} got=csrf_{random.randint(1000, 9999)} sessionId=sess_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Path traversal attempt blocked path={random.choice(['../../../etc/passwd', '..\\..\\Windows\\System32', '%2e%2e%2f%2e%2e%2fetc'])}"
    ),
    lambda: (
        f"File upload contains malicious content detected={random.choice(['EICAR_TEST', 'PHP_WEBSHELL', 'MALWARE_DROPPER'])} fileName={random.choice(['shell.php', 'evil.exe', 'malware.jar', 'test.jsp'])}"
    ),
    lambda: (
        f"Privilege escalation attempt userId={random.randint(1000, 99999)} attemptedRole={random.choice(['ADMIN', 'SUPER_ADMIN', 'MODERATOR'])} currentRole={random.choice(['USER', 'GUEST'])}"
    ),
    lambda: (
        f"Request from blocked country origin={random.choice(['RU', 'CN', 'KP', 'IR', 'MM', 'CU'])} ip={random.randint(1, 223)}.{random.randint(1, 223)}.{random.randint(1, 223)}.{random.randint(1, 254)}"
    ),
    lambda: (
        f"SSRF attempt blocked url=http://{random.randint(10, 99)}.{random.randint(10, 99)}.{random.randint(1, 10)}.{random.randint(1, 254)}:{random.randint(22, 9200)}/internal"
    ),
    lambda: (
        f"Mass assignment attack prevented field={random.choice(['role', 'isAdmin', 'balance', 'creditLimit'])} value={random.choice(['ADMIN', 'true', '999999', 'unlimited'])}"
    ),
]

msgs["IOException"] = [
    lambda: (
        f"Disk write failed no space left on device mount={random.choice(['/data', '/var/log', '/tmp', '/app/logs'])} available={random.randint(0, 100)}MB"
    ),
    lambda: (
        f"File not found exception path=/app/{random.choice(['config', 'data', 'logs', 'uploads'])}/file_{random.randint(1000, 9999)}.{random.choice(['json', 'csv', 'xml', 'log'])}"
    ),
    lambda: (
        f"Stream closed unexpectedly while reading from {random.choice(['HTTP input', 'socket', 'file channel', 'pipe'])} bytesRead={random.randint(0, 1024)} expected={random.randint(2048, 65536)}"
    ),
    lambda: (
        f"Network socket closed before completing handshake localPort={random.randint(10000, 60000)} remote=10.0.{random.randint(10, 99)}.{random.randint(1, 254)}:{random.randint(3306, 9200)}"
    ),
    lambda: (
        f"Failed to create temp file in directory={random.choice(['/tmp', '/var/tmp', '/app/temp'])} reason={random.choice(['permission denied', 'read-only fs', 'disk quota exceeded'])}"
    ),
    lambda: (
        f"Pipe broken between writer and reader thread={random.choice(['main', 'worker-1', 'io-pool'])} bufferSize={random.randint(1024, 65536)}"
    ),
    lambda: (
        f"Interrupted I/O operation while reading from {random.choice(['stdin', 'file', 'network', 'pipe'])} threadId={random.randint(1, 999)}"
    ),
    lambda: (
        f"File lock acquisition failed path=/app/{random.choice(['data', 'indexes', 'locks'])}/lock_{random.randint(1, 999)}.lck heldBy=process_{random.randint(1000, 9999)}"
    ),
    lambda: (
        f"Zip file corrupted at entry={random.choice(['data.csv', 'images.zip', 'backup.tar.gz', 'dump.sql'])} offset={random.randint(1000, 99999)}"
    ),
    lambda: (
        f"Maximum open file descriptors reached limit={random.randint(1024, 65536)} current={random.randint(1024, 65536)} process=java"
    ),
    lambda: (
        f"Character encoding error input=0x{random.randint(128, 255):02X}{random.randint(128, 255):02X}{random.randint(128, 255):02X} expected=UTF-8"
    ),
    lambda: (
        f"Symlink loop detected path={random.choice(['/app/data/../data/../data', '/logs/../logs/../logs'])} depth={random.randint(10, 100)}"
    ),
    lambda: (
        f"File already exists but cannot overwrite path=/app/{random.choice(['config', 'data'])}/file_{random.randint(1000, 9999)}.{random.choice(['lock', 'tmp', 'pid'])}"
    ),
]

TOTAL = 500000
OUTPUT = "D:/alouette-AI/error_logs_500k.csv"

counts = {e: 0 for e in exceptions}
with open(OUTPUT, "w", encoding="utf-8") as f:
    for i in range(TOTAL):
        level = random.choice(levels)
        ts = base_time + timedelta(seconds=random.randint(0, 86399))
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        svc = random.choice(services)
        exc = random.choice(exceptions)
        counts[exc] += 1
        msg = random.choice(msgs[exc])()
        f.write(f"{level},{ts_str},{svc},{exc},{msg}\n")
        if (i + 1) % 50000 == 0:
            print(f"  ... {i + 1}/{TOTAL} lines written")

import os

size = os.path.getsize(OUTPUT)
print(f"\nDone - {TOTAL} lines written to {OUTPUT}")
print(f"File size: {size / 1024 / 1024:.1f} MB")
print(f"\nException distribution:")
for e, c in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {e}: {c} ({c * 100 / TOTAL:.1f}%)")

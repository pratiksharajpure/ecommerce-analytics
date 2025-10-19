-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INT PRIMARY KEY AUTO_INCREMENT,
    order_id INT NOT NULL,
    transaction_date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    amount DECIMAL(10, 2) NOT NULL,
    payment_method VARCHAR(50),
    status VARCHAR(20) DEFAULT 'pending',
    currency VARCHAR(3) DEFAULT 'USD',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_order_id (order_id),
    INDEX idx_transaction_date (transaction_date),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Populate with data from orders (safe version - only uses columns that exist)
INSERT INTO transactions (order_id, transaction_date, amount, status)
SELECT 
    order_id,
    order_date as transaction_date,
    total_amount as amount,
    CASE 
        WHEN status = 'completed' THEN 'success'
        WHEN status = 'cancelled' THEN 'failed'
        WHEN status = 'pending' THEN 'pending'
        ELSE 'success'
    END as status
FROM orders
WHERE NOT EXISTS (
    SELECT 1 FROM transactions WHERE transactions.order_id = orders.order_id
)
LIMIT 10000;
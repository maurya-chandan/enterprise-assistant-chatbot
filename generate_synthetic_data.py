"""
Generates a synthetic e-commerce dataset for the Customer Service RAG demo:
  - customers.csv   (~1,200 rows)
  - products.csv    (~300 rows)
  - orders.csv      (5,000 rows)  <- the "5,000 records" the project is built around
  - tracking.csv     (~4,300 rows, one per shipped/delivered/returned order)

This is reference/dimension data (customers, products) plus a fact table
(orders) with a dependent table (tracking) - a small, realistic stand-in for
what would normally come from a CRM + order management + WMS + carrier API.

Re-running this script regenerates all four files from scratch (it does not
append). A fixed random seed keeps the output reproducible; remove the
`random.seed(...)` line below if you want different data on every run.
"""

import os
import csv
import json
import random
from datetime import date, timedelta

random.seed(42)

DATA_DIR = "./data"

NUM_CUSTOMERS = 1200
NUM_PRODUCTS = 300
NUM_ORDERS = 5000

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Priya", "Arjun", "Wei", "Mei", "Carlos",
    "Sofia", "Yuki", "Hiro", "Fatima", "Omar", "Ivan", "Elena", "Liam", "Noah",
    "Emma", "Olivia", "Ava", "Sophia", "Mia", "Amelia", "Lucas", "Ethan",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Patel", "Kumar", "Nakamura", "Kim", "Chen", "Novak", "Ivanov", "Haddad",
]
CITIES_STATES = [
    ("Austin", "TX"), ("Denver", "CO"), ("Seattle", "WA"), ("Chicago", "IL"),
    ("Atlanta", "GA"), ("Phoenix", "AZ"), ("Boston", "MA"), ("Portland", "OR"),
    ("Miami", "FL"), ("Columbus", "OH"), ("Raleigh", "NC"), ("San Diego", "CA"),
    ("Minneapolis", "MN"), ("Nashville", "TN"), ("Salt Lake City", "UT"),
]
LOYALTY_TIERS = ["Bronze", "Bronze", "Bronze", "Silver", "Silver", "Gold", "Platinum"]

PRODUCT_CATEGORIES = {
    "Electronics": (["Wireless", "Bluetooth", "USB-C", "Smart", "Portable", "HD"],
                     ["Mouse", "Keyboard", "Speaker", "Charger", "Headphones", "Webcam", "Monitor", "Hub"],
                     (9.99, 249.99)),
    "Home & Kitchen": (["Stainless", "Ceramic", "Non-stick", "Electric", "Compact", "Deluxe"],
                        ["Kettle", "Blender", "Cutting Board", "Toaster", "Air Fryer", "Cookware Set", "Mug", "Pan"],
                        (7.99, 179.99)),
    "Apparel": (["Cotton", "Slim-Fit", "Fleece", "Waterproof", "Merino", "Classic"],
                ["T-Shirt", "Jacket", "Hoodie", "Socks", "Jeans", "Cap", "Scarf"],
                (5.99, 129.99)),
    "Sports & Outdoors": (["Adjustable", "Foldable", "Insulated", "Lightweight", "All-Terrain"],
                           ["Yoga Mat", "Water Bottle", "Tent", "Backpack", "Dumbbell Set", "Bike Helmet"],
                           (8.99, 219.99)),
    "Books": (["Illustrated", "Revised", "Collector's", "Pocket"],
              ["Cookbook", "Novel", "Field Guide", "Journal", "Atlas"],
              (6.99, 39.99)),
    "Beauty": (["Organic", "Fragrance-Free", "Hydrating", "Matte", "Travel-Size"],
               ["Moisturizer", "Shampoo", "Lip Balm", "Sunscreen", "Face Wash"],
               (4.99, 59.99)),
    "Grocery": (["Organic", "Gluten-Free", "Single-Origin", "Family-Size", "No-Sugar"],
                ["Coffee Beans", "Pasta", "Trail Mix", "Olive Oil", "Granola"],
                (3.49, 34.99)),
}
WAREHOUSES = ["Warehouse A - Dallas, TX", "Warehouse B - Reno, NV",
              "Warehouse C - Columbus, OH", "Warehouse D - Newark, NJ"]

CARRIERS = ["FedEx", "UPS", "USPS", "DHL"]
ORDER_STATUS_WEIGHTS = [
    ("Delivered", 50), ("Shipped", 15), ("Out for Delivery", 5),
    ("Processing", 12), ("Delayed", 4), ("Cancelled", 7), ("Returned", 7),
]
TRACKING_STATUS_BY_ORDER_STATUS = {
    "Shipped": "In Transit",
    "Out for Delivery": "Out for Delivery",
    "Delivered": "Delivered",
    "Delayed": "Delayed - carrier exception",
    "Returned": "Returned to Sender",
}
PAYMENT_METHODS = ["Visa", "Mastercard", "Amex", "PayPal", "Store Credit", "Apple Pay"]


def weighted_choice(pairs):
    options, weights = zip(*pairs)
    return random.choices(options, weights=weights, k=1)[0]


def random_date(start: date, end: date) -> date:
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta_days, 0)))


def generate_customers(n):
    rows = []
    today = date.today()
    for i in range(1, n + 1):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        city, state = random.choice(CITIES_STATES)
        customer_id = f"CUST{i:05d}"
        rows.append({
            "customer_id": customer_id,
            "first_name": first,
            "last_name": last,
            "email": f"{first.lower()}.{last.lower()}{i}@example.com",
            "phone": f"{random.randint(200,989)}-{random.randint(200,989)}-{random.randint(1000,9999)}",
            "address": f"{random.randint(100,9999)} {random.choice(['Maple','Oak','Sunset','2nd','Main','Cedar','Pine'])} St",
            "city": city,
            "state": state,
            "zip": f"{random.randint(10000,99999)}",
            "join_date": random_date(today - timedelta(days=3*365), today - timedelta(days=1)).isoformat(),
            "loyalty_tier": random.choice(LOYALTY_TIERS),
        })
    return rows


def generate_products(n):
    rows = []
    categories = list(PRODUCT_CATEGORIES.keys())
    for i in range(1, n + 1):
        category = categories[i % len(categories)]
        adjectives, nouns, price_range = PRODUCT_CATEGORIES[category]
        name = f"{random.choice(adjectives)} {random.choice(nouns)}"
        price = round(random.uniform(*price_range), 2)
        rows.append({
            "product_id": f"PROD{i:04d}",
            "product_name": name,
            "category": category,
            "unit_price": price,
            "stock_quantity": random.randint(0, 2000),
            "warehouse": random.choice(WAREHOUSES),
        })
    return rows


def generate_orders_and_tracking(n, customers, products):
    order_rows = []
    tracking_rows = []
    today = date.today()

    # Skew repeat business toward a subset of customers (roughly 80/20).
    frequent_customers = random.sample(customers, k=max(1, len(customers) // 5))

    for i in range(1, n + 1):
        order_id = f"ORD{100000 + i}"
        customer = random.choice(frequent_customers) if random.random() < 0.7 else random.choice(customers)
        order_date = random_date(today - timedelta(days=365), today)

        num_items = random.choices([1, 2, 3, 4], weights=[55, 25, 12, 8])[0]
        chosen_products = random.sample(products, k=min(num_items, len(products)))
        items = []
        order_total = 0.0
        for p in chosen_products:
            qty = random.randint(1, 5)
            unit_price_at_order = p["unit_price"]
            items.append(f"{p['product_id']}:{qty}:{unit_price_at_order}")
            order_total += qty * unit_price_at_order

        status = weighted_choice(ORDER_STATUS_WEIGHTS)

        order_rows.append({
            "order_id": order_id,
            "customer_id": customer["customer_id"],
            "order_date": order_date.isoformat(),
            "items": "|".join(items),
            "order_total": round(order_total, 2),
            "payment_method": random.choice(PAYMENT_METHODS),
            "order_status": status,
        })

        if status in TRACKING_STATUS_BY_ORDER_STATUS:
            shipped_date = order_date + timedelta(days=random.randint(1, 3))
            est_delivery = shipped_date + timedelta(days=random.randint(2, 9))
            city, state = random.choice(CITIES_STATES)
            tracking_rows.append({
                "order_id": order_id,
                "carrier": random.choice(CARRIERS),
                "tracking_number": f"{random.choice(CARRIERS)[:2].upper()}{random.randint(10**9, 10**10-1)}",
                "shipped_date": shipped_date.isoformat(),
                "estimated_delivery": est_delivery.isoformat(),
                "current_status": TRACKING_STATUS_BY_ORDER_STATUS[status],
                "last_location": f"{city}, {state}",
            })

    return order_rows, tracking_rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Generating {NUM_CUSTOMERS} customers...")
    customers = generate_customers(NUM_CUSTOMERS)
    write_csv(os.path.join(DATA_DIR, "customers.csv"), customers, list(customers[0].keys()))

    print(f"Generating {NUM_PRODUCTS} products...")
    products = generate_products(NUM_PRODUCTS)
    write_csv(os.path.join(DATA_DIR, "products.csv"), products, list(products[0].keys()))

    print(f"Generating {NUM_ORDERS} orders (with tracking for shipped/delivered/returned orders)...")
    orders, tracking = generate_orders_and_tracking(NUM_ORDERS, customers, products)
    write_csv(os.path.join(DATA_DIR, "orders.csv"), orders, list(orders[0].keys()))
    write_csv(os.path.join(DATA_DIR, "tracking.csv"), tracking, list(tracking[0].keys()))

    print(
        f"Done. Wrote {len(customers)} customers, {len(products)} products, "
        f"{len(orders)} orders, {len(tracking)} tracking records to {DATA_DIR}/"
    )


if __name__ == "__main__":
    main()

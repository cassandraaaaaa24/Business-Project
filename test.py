
import tkinter as tk

from tkinter import ttk, messagebox

from datetime import datetime

import collections
 
class TransactionApp:

    def __init__(self, root):

        self.root = root

        self.root.title("Customer Transactions Tracker")

        self.root.geometry("750x500")

        self.root.configure(bg="#ffe6f0")  # soft pink background
 
        # Apply style

        style = ttk.Style()

        style.theme_use("clam")
 
        # Pink theme styling

        style.configure("TButton",

                        font=("Arial", 10, "bold"),

                        background="#ff99cc",

                        foreground="black",

                        padding=6)

        style.map("TButton",

                  background=[("active", "#ff66b2")])
 
        style.configure("TLabel",

                        font=("Arial", 10),

                        background="#ffe6f0")
 
        style.configure("Treeview",

                        background="#fff0f5",

                        fieldbackground="#fff0f5",

                        foreground="black")

        style.configure("Treeview.Heading",

                        font=("Arial", 10, "bold"),

                        background="#ff99cc",

                        foreground="black")
 
        # --- Data storage ---

        self.transactions = []
 
        # --- Input fields ---

        frame = ttk.LabelFrame(root, text="Add Transaction", padding=10)

        frame.pack(pady=10, fill="x")

        frame.configure(style="TLabelframe")
 
        ttk.Label(frame, text="Customer ID").grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.customer_id = ttk.Entry(frame)

        self.customer_id.grid(row=0, column=1, padx=5, pady=5)
 
        ttk.Label(frame, text="Item Name").grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.item_name = ttk.Entry(frame)

        self.item_name.grid(row=1, column=1, padx=5, pady=5)
 
        ttk.Label(frame, text="Category").grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.category = ttk.Entry(frame)

        self.category.grid(row=2, column=1, padx=5, pady=5)
 
        ttk.Label(frame, text="Price").grid(row=3, column=0, padx=5, pady=5, sticky="w")

        self.price = ttk.Entry(frame)

        self.price.grid(row=3, column=1, padx=5, pady=5)
 
        ttk.Button(frame, text="Add Transaction", command=self.add_transaction).grid(row=4, column=0, columnspan=2, pady=10)
 
        # --- Transaction table ---

        table_frame = ttk.Frame(root)

        table_frame.pack(pady=10, fill="both", expand=True)
 
        self.tree = ttk.Treeview(table_frame, columns=("Date", "Customer ID", "Item", "Category", "Price"), show="headings")

        self.tree.pack(side="left", fill="both", expand=True)
 
        for col in ("Date", "Customer ID", "Item", "Category", "Price"):

            self.tree.heading(col, text=col)

            self.tree.column(col, width=120)
 
        # Add scrollbar

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)

        self.tree.configure(yscroll=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
 
        # --- Trend analysis button ---

        ttk.Button(root, text="📊 Predict Sales Trends", command=self.predict_trends).pack(pady=10)
 
        # --- Status bar ---

        self.status = ttk.Label(root, text="Ready", relief="sunken", anchor="w")

        self.status.pack(side="bottom", fill="x")
 
    def add_transaction(self):

        try:

            price = float(self.price.get())

        except ValueError:

            messagebox.showerror("Error", "Price must be a number")

            return
 
        transaction = {

            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),

            "customer_id": self.customer_id.get(),

            "item": self.item_name.get(),

            "category": self.category.get(),

            "price": price

        }

        self.transactions.append(transaction)
 
        self.tree.insert("", "end", values=(transaction["date"], transaction["customer_id"],

                                            transaction["item"], transaction["category"], transaction["price"]))
 
        # Clear inputs

        self.customer_id.delete(0, tk.END)

        self.item_name.delete(0, tk.END)

        self.category.delete(0, tk.END)

        self.price.delete(0, tk.END)
 
        self.status.config(text="Transaction added successfully ✅")
 
    def predict_trends(self):

        if not self.transactions:

            messagebox.showinfo("Trends", "No transactions yet.")

            return
 
        categories = collections.Counter(t["category"] for t in self.transactions)

        items = collections.Counter(t["item"] for t in self.transactions)

        total_sales = sum(t["price"] for t in self.transactions)
 
        trend_msg = "📊 Sales Trend Analysis:\n\n"

        trend_msg += f"Total Sales: ${total_sales:.2f}\n\n"

        trend_msg += "Top Categories:\n"

        for cat, count in categories.most_common(3):

            trend_msg += f" - {cat}: {count} purchases\n"

        trend_msg += "\nTop Items:\n"

        for item, count in items.most_common(3):

            trend_msg += f" - {item}: {count} purchases\n"
 
        messagebox.showinfo("Sales Trends", trend_msg)
 
# --- Run the app ---

if __name__ == "__main__":

    root = tk.Tk()

    app = TransactionApp(root)

    root.mainloop()

 
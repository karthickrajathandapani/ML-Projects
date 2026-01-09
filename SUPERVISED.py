import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------------------
# Step 1: Create Dataset
# ---------------------------
data = {
    'Income': [25, 30, 45, 50, 60, 35, 80, 90, 20, 70],
    'CreditScore': [550, 600, 650, 700, 750, 620, 800, 820, 500, 780],
    'LoanApproved': [0, 0, 1, 1, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Income', 'CreditScore']]
y = df['LoanApproved']

# ---------------------------
# Step 2: Train ML Model
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------
# Step 3: Prediction Function
# ---------------------------
def predict_loan():
    try:
        income = float(entry_income.get())
        credit_score = int(entry_credit.get())

        result = model.predict([[income, credit_score]])

        if result[0] == 1:
            messagebox.showinfo("Result", "✅ Loan kedicherum")
        else:
            messagebox.showwarning("Result", "❌ Loan la ilaa velia po da")

    except ValueError:
        messagebox.showerror("Error", "crt taa na value kudu da diiii")

# ---------------------------
# Step 4: Create GUI Window
# ---------------------------
root = tk.Tk()
root.title("Loan Approval Prediction System")
root.geometry("1000x1000")
root.resizable(False, False)

# ---------------------------
# Step 5: GUI Components
# ---------------------------
tk.Label(root, text="Loan Approval Prediction", font=("Arial", 16, "bold")).pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=10)

tk.Label(frame, text="Monthly Income (in thousands):").grid(row=0, column=0, pady=5, sticky="w")
entry_income = tk.Entry(frame)
entry_income.grid(row=0, column=1)

tk.Label(frame, text="Credit Score:").grid(row=1, column=0, pady=5, sticky="w")
entry_credit = tk.Entry(frame)
entry_credit.grid(row=1, column=1)

tk.Button(
    root,
    text="Check Loan Status",
    command=predict_loan,
    bg="green",
    fg="white",
    font=("Arial", 12)
).pack(pady=20)

tk.Label(root, text="Supervised Learning | Logistic Regression", font=("Arial", 9)).pack(side="bottom")

# ---------------------------
# Step 6: Run Application
# ---------------------------
root.mainloop()

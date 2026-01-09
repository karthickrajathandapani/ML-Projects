class MonthlyExpense:
    def __init__(self, salary):
        self.salary = salary
        self.expenses = {}

    def add_expense(self):
        while True:
            purpose = input("\nEnter expense purpose (or 'done' to finish): ")

            if purpose.lower() == "done": #lower() converts user input into lowercase to avoid case-sensitive errors.
                break

            amount = int(input("Enter money spent: "))
            self.expenses[purpose] = amount #"Rent": 12000

    def calculate_total(self):
        return sum(self.expenses.values()) #sum the numbers in the set

    def show_summary(self):
        total_spent = self.calculate_total()
        remaining = self.salary - total_spent

        print("\n📊 MONTHLY EXPENSE SUMMARY")
        print("-" * 30)

        for purpose, amount in self.expenses.items(): #.items() allows iterating through both keys and values of a dictionary at the same time. like "rent" 1000
            print(f"{purpose} : ₹{amount}")

        print("-" * 30)
        print("Total Spent : ₹", total_spent)
        print("Remaining Salary : ₹", remaining)


# -------- Main Program --------
salary = int(input("Enter your monthly salary: "))

expense = MonthlyExpense(salary)
expense.add_expense()
expense.show_summary()

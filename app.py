import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ====================================================== Load Local CSS ===============================================================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# =================================================== UI CONFIGURATION ========================================================
st.set_page_config(
    page_title="College Expense Tracker",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================== CUSTOM CSS ==============================================================

st.markdown("""
<style>
.metric-card {
    background: white;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.main-header {
    font-size: 2.8rem;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    background: #f0f2f6;
    border-radius: 10px 10px 0px 0px;
    padding: 10px 20px;
    border: none;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1f77b4;
    color: white;
}
.dataframe {
    border-radius: 10px;
    overflow: hidden;
}
.dataframe th {
    background: #1f77b4;
    color: white;
    text-align: center;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================================================== HEADER ===========================================================

col1, col2, col3 = st.columns([0.8, 2.4, 0.8])
with col2:
    st.markdown('<h1 class="main-header">💸 College Expense Tracker</h1>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color: #555;'>Smart Budgeting App for Smart Students 📚</h3>", unsafe_allow_html=True)

# ====================================================== DATA MANAGEMENT =====================================================

if 'expenses' not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns=[
        'Date', 'Category', 'Description', 'Amount', 'Payment_Method'
    ])
if 'budget' not in st.session_state:
    st.session_state.budget = {
        'Food': 5000, 'Transport': 2000, 'Books': 3000,
        'Fees': 10000, 'Entertainment': 1500, 'Other': 2000
    }
if 'savings_goals' not in st.session_state:
    st.session_state.savings_goals = {}

# ============================================== LOADING OF THE DATASET ============================================

@st.cache_data
def load_kaggle_data():
    try:
        data = pd.read_csv('expenses.csv')

        column_mapping = {}
        amount_keywords = ['amount', 'price', 'cost', 'expense', 'money']
        for col in data.columns:
            if any(keyword in col.lower() for keyword in amount_keywords):
                column_mapping[col] = 'Amount'
                break

        category_keywords = ['category', 'type', 'expense_type']
        for col in data.columns:
            if any(keyword in col.lower() for keyword in category_keywords):
                column_mapping[col] = 'Category'
                break

        if column_mapping:
            data = data.rename(columns=column_mapping)

        if 'Amount' not in data.columns:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data['Amount'] = data[numeric_cols[0]]
            else:
                data['Amount'] = np.random.randint(50, 5000, len(data))

        if 'Category' not in data.columns:
            data['Category'] = np.random.choice(
                ['Food', 'Transport', 'Books', 'Fees', 'Entertainment', 'Other'],
                len(data)
            )

        if 'Date' not in data.columns:
            data['Date'] = pd.date_range(
                start='2024-01-01', periods=len(data), freq='D'
            )

        if 'Description' not in data.columns:
            data['Description'] = ['Expense ' + str(i) for i in range(len(data))]

        if 'Payment_Method' not in data.columns:
            data['Payment_Method'] = np.random.choice(
                ['Cash', 'UPI', 'Card', 'Online Transfer'],
                len(data)
            )

        return data

    except FileNotFoundError:
        sample_data = pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'Category': np.random.choice(
                ['Food', 'Transport', 'Books', 'Fees', 'Entertainment', 'Other'], 100
            ),
            'Description': ['Expense ' + str(i) for i in range(100)],
            'Amount': np.random.randint(50, 5000, 100),
            'Payment_Method': np.random.choice(
                ['Cash', 'UPI', 'Card', 'Online Transfer'], 100
            )
        })
        return sample_data

kaggle_data = load_kaggle_data()

# ================================================ SIDEBAR NAVIGATION ==================================================

with st.sidebar:
    st.markdown("## 🧭 Navigation")

    page = st.radio(
        "Choose a section:",
        ["📊 Dashboard", "➕ Add Expense", "🎯 Budget",
         "📈 Analytics", "🔮 Predictions", "💰 Savings", "💾 Data"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if not st.session_state.expenses.empty:
        st.markdown("### 📈 Quick Stats")

        total = st.session_state.expenses['Amount'].sum()
        st.metric("Total Spent", f"₹{total:,.0f}")

        food_spent = st.session_state.expenses[
            st.session_state.expenses['Category'] == 'Food'
        ]['Amount'].sum()

        budget_used = (
            (food_spent / st.session_state.budget['Food']) * 100
            if st.session_state.budget['Food'] > 0 else 0
        )

        st.progress(min(int(budget_used), 100), text="Food Budget Used")

        st.markdown("### 📝 Recent Activity")
        recent = st.session_state.expenses.tail(3)
        for _, expense in recent.iterrows():
            st.text(f"₹{expense['Amount']} - {expense['Category']}")

# ============================================= DASHBOARD PAGE ==============================================

if page == "📊 Dashboard":

    st.markdown("## 📊 Financial Dashboard")

    if st.button("🔄 Load Dataset", use_container_width=True):
        st.session_state.expenses = kaggle_data
        st.success("✅ Dataset loaded successfully!")

    if not st.session_state.expenses.empty:

        st.markdown("### 📊 Quick Overview")
        col1, col2, col3, col4 = st.columns(4)

        total_expenses = st.session_state.expenses['Amount'].sum()
        avg_daily = st.session_state.expenses.groupby('Date')['Amount'].sum().mean()
        common_category = st.session_state.expenses['Category'].mode()[0]
        total_records = len(st.session_state.expenses)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💰 Total Spent", f"₹{total_expenses:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📅 Daily Average", f"₹{avg_daily:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🏆 Top Category", common_category)
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📋 Total Records", f"{total_records:,}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 📊 Visual Analytics")

        tab_option = st.radio(
            "Select a visualization:",
            ["📈 Spending Trends", "🏷️ Categories", "💳 Payment Methods"],
            horizontal=True
        )

        if tab_option == "📈 Spending Trends":
            st.markdown("### 📅 Last 30 Days Spending")

            daily_expenses = st.session_state.expenses.groupby('Date')['Amount'] \
                .sum().reset_index().sort_values('Date')

            daily_expenses = daily_expenses.tail(30)

            fig = px.bar(
                daily_expenses,
                x='Date', y='Amount',
                text_auto=True,
                title='Recent Spending Trend (Last 30 Days)'
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        elif tab_option == "🏷️ Categories":
            st.markdown("### 🏷️ Spending by Category")

            category_totals = st.session_state.expenses.groupby('Category')['Amount'] \
                .sum().reset_index().sort_values('Amount', ascending=True)

            fig = px.bar(
                category_totals,
                x='Amount', y='Category',
                orientation='h',
                text_auto=True,
                title='Category-wise Spending'
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        elif tab_option == "💳 Payment Methods":
            st.markdown("### 💳 Payment Methods Used")

            payment_counts = st.session_state.expenses['Payment_Method'] \
                .value_counts().reset_index()

            payment_counts.columns = ['Payment_Method', 'Count']

            fig = px.bar(
                payment_counts,
                x='Payment_Method', y='Count',
                text_auto=True,
                title='Preferred Payment Methods'
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("## 💳 Recent Transactions")

        recent_data = st.session_state.expenses.tail(8).copy()
        recent_data['Amount'] = recent_data['Amount'].apply(
            lambda x: f"₹{x:,.0f}"
        )
        recent_data['Date'] = recent_data['Date'].astype(str)

        st.dataframe(
            recent_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": "📅 Date",
                "Category": "🏷️ Category",
                "Description": "📝 Description",
                "Amount": "💰 Amount",
                "Payment_Method": "💳 Payment"
            }
        )

    else:
        st.info("🌟 No expenses recorded yet. Load the dataset or add some expenses to get started!")

# ============================================================= ADD EXPENSE PAGE =================================================

elif page == "➕ Add Expense":

    st.markdown("## ➕ Add New Expense")

    with st.form("expense_form", clear_on_submit=True):

        st.markdown("### 💰 Expense Details")

        col1, col2 = st.columns(2)

        with col1:
            date = st.date_input("📅 Date", datetime.now())
            category = st.selectbox(
                "🏷️ Category",
                ["Food", "Transport", "Books", "Fees", "Entertainment", "Other"]
            )
            amount = st.number_input(
                "💰 Amount (₹)",
                min_value=0.0, format="%.2f", step=100.0
            )

        with col2:
            description = st.text_input(
                "📝 Description",
                placeholder="Lunch, Bus fare, Books, etc."
            )
            payment_method = st.selectbox(
                "💳 Payment Method",
                ["Cash", "UPI", "Card", "Online Transfer"]
            )

        submitted = st.form_submit_button(
            "💾 Save Expense",
            use_container_width=True
        )

        if submitted:
            if amount <= 0:
                st.error("❌ Amount must be greater than 0")
            elif not description.strip():
                st.error("❌ Please enter a description")
            else:
                new_expense = pd.DataFrame({
                    'Date': [date],
                    'Category': [category],
                    'Description': [description],
                    'Amount': [amount],
                    'Payment_Method': [payment_method]
                })

                st.session_state.expenses = pd.concat(
                    [st.session_state.expenses, new_expense],
                    ignore_index=True
                )

                st.success("✅ Expense added successfully!")

# ======================================================= BUDGET PAGE =============================================================

elif page == "🎯 Budget":

    st.markdown("## 🎯 Budget Management")

    st.markdown("### 💵 Set Monthly Budgets")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)

        st.session_state.budget['Food'] = st.number_input(
            "🍕 Food Budget (₹)",
            value=st.session_state.budget['Food'], step=500
        )
        st.session_state.budget['Transport'] = st.number_input(
            "🚗 Transport Budget (₹)",
            value=st.session_state.budget['Transport'], step=500
        )
        st.session_state.budget['Books'] = st.number_input(
            "📚 Books Budget (₹)",
            value=st.session_state.budget['Books'], step=500
        )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)

        st.session_state.budget['Fees'] = st.number_input(
            "🎓 Fees Budget (₹)",
            value=st.session_state.budget['Fees'], step=1000
        )
        st.session_state.budget['Entertainment'] = st.number_input(
            "🎬 Entertainment Budget (₹)",
            value=st.session_state.budget['Entertainment'], step=500
        )
        st.session_state.budget['Other'] = st.number_input(
            "📦 Other Budget (₹)",
            value=st.session_state.budget['Other'], step=500
        )

        st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state.expenses.empty:

        st.markdown("---")
        st.markdown("### 📊 Budget vs Actual Spending")

        categories = list(st.session_state.budget.keys())

        budget_values = [
            st.session_state.budget[cat] for cat in categories
        ]
        actual_values = [
            st.session_state.expenses[
                st.session_state.expenses['Category'] == cat
            ]['Amount'].sum()
            for cat in categories
        ]

        fig = go.Figure(data=[
            go.Bar(name='💰 Budget', x=categories, y=budget_values, marker_color='#2ecc71'),
            go.Bar(name='💸 Actual', x=categories, y=actual_values, marker_color='#e74c3c')
        ])

        fig.update_layout(
            title='Budget vs Actual Spending',
            height=500,
            barmode='group'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Budget Status")

        budget_status = []

        for category in categories:
            budget_amount = st.session_state.budget[category]
            actual_amount = st.session_state.expenses[
                st.session_state.expenses['Category'] == category
            ]['Amount'].sum()
            remaining = budget_amount - actual_amount
            usage_percent = (
                (actual_amount / budget_amount) * 100
                if budget_amount else 0
            )

            status = "✅ Within Budget" if remaining >= 0 else "❌ Over Budget"

            budget_status.append({
                'Category': category,
                'Budget (₹)': f"₹{budget_amount:,}",
                'Actual (₹)': f"₹{actual_amount:,}",
                'Remaining (₹)': f"₹{remaining:,}",
                'Usage (%)': f"{usage_percent:.1f}%",
                'Status': status
            })

        budget_df = pd.DataFrame(budget_status)

        st.dataframe(
            budget_df,
            use_container_width=True,
            hide_index=True
        )

# ====================================================== ANALYTICS PAGE =========================================================

elif page == "📈 Analytics":

    st.markdown("## 📈 Advanced Analytics")

    if not st.session_state.expenses.empty:

        st.markdown("### 📅 Monthly Spending Analysis")

        expenses_copy = st.session_state.expenses.copy()
        expenses_copy['Month'] = pd.to_datetime(
            expenses_copy['Date']
        ).dt.to_period('M')

        monthly_totals = expenses_copy.groupby('Month')['Amount'].sum().reset_index()
        monthly_totals['Month'] = monthly_totals['Month'].astype(str)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                monthly_totals, x='Month', y='Amount',
                title='Monthly Spending Trend', markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                monthly_totals, x='Month', y='Amount',
                title='Monthly Spending Comparison'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🏆 Category-wise Analysis")

        category_stats = st.session_state.expenses.groupby('Category').agg({
            'Amount': ['sum', 'count', 'mean', 'max']
        }).round(2)

        category_stats.columns = [
            'Total Amount', 'Transaction Count', 'Average Amount', 'Max Amount'
        ]

        category_stats = category_stats.sort_values(
            'Total Amount', ascending=False
        )

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(category_stats, use_container_width=True)

        with col2:
            expenses_copy = st.session_state.expenses.copy()
            expenses_copy['DayOfWeek'] = pd.to_datetime(
                expenses_copy['Date']
            ).dt.day_name()

            day_order = ['Monday', 'Tuesday', 'Wednesday',
                         'Thursday', 'Friday', 'Saturday', 'Sunday']

            weekly_pattern = expenses_copy.groupby('DayOfWeek')['Amount'].sum().reindex(day_order)

            fig = px.bar(
                x=weekly_pattern.index, y=weekly_pattern.values,
                title='Spending by Day of Week',
                labels={'x': 'Day', 'y': 'Amount'}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 💳 Payment Method Insights")

        payment_analysis = st.session_state.expenses.groupby('Payment_Method').agg({
            'Amount': ['sum', 'count', 'mean']
        }).round(2)

        payment_analysis.columns = [
            'Total Amount', 'Transaction Count', 'Average Amount'
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(payment_analysis, use_container_width=True)

        with col2:
            fig = px.pie(
                payment_analysis,
                values='Total Amount',
                names=payment_analysis.index,
                title='Payment Method Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("📊 No data available for analytics. Add expenses or load data first.")

# ===================================================== SAVINGS PAGE ========================================================

elif page == "💰 Savings":

    st.markdown("## 💰 Savings Goals & Tips")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Set Savings Goals")

        with st.form("savings_goal"):
            goal_name = st.text_input("Goal Name", placeholder="New Laptop, Trip, etc.")
            target_amount = st.number_input("Target Amount (₹)", min_value=0, step=1000)
            timeframe = st.selectbox("Timeframe", ["1 month", "3 months", "6 months", "1 year"])

            if st.form_submit_button("🎯 Set Goal"):
                if goal_name and target_amount > 0:
                    st.session_state.savings_goals[goal_name] = {
                        'target': target_amount,
                        'timeframe': timeframe,
                        'created': datetime.now()
                    }
                    st.success(f"✅ Goal '{goal_name}' set successfully!")

    with col2:
        st.markdown("### 📊 Current Goals & Progress")

        if st.session_state.savings_goals:

            for goal, details in st.session_state.savings_goals.items():

                total_expenses = st.session_state.expenses['Amount'].sum()
                monthly_income = 20000
                current_savings = max(0, monthly_income - total_expenses)

                target = details['target']
                progress = min((current_savings / target) * 100, 100) if target > 0 else 0

                st.markdown(f"""
                **{goal}**
                - 🎯 Target: ₹{target:,}
                - ⏰ Timeframe: {details['timeframe']}
                - 💰 Current Savings: ₹{current_savings:,.0f}
                """)

                st.progress(progress / 100, text=f"Progress: {progress:.1f}%")

                created_date = details['created']
                timeframe_days = {
                    "1 month": 30,
                    "3 months": 90,
                    "6 months": 180,
                    "1 year": 365
                }.get(details['timeframe'], 30)

                days_passed = (datetime.now() - created_date).days
                days_remaining = max(0, timeframe_days - days_passed)

                st.caption(
                    f"⏳ {days_remaining} days remaining • Created: {created_date.strftime('%b %d, %Y')}"
                )
                st.markdown("---")

            total_target = sum(details['target'] for details in st.session_state.savings_goals.values())
            st.metric("🎯 Total Goals Target", f"₹{total_target:,}")
            st.metric("💰 Current Monthly Savings", f"₹{current_savings:,}")

        else:
            st.info("🎯 No savings goals set yet. Add one above!")

    st.markdown("---")
    st.markdown("### 💡 Smart Savings Strategies")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🍕 Food & Dining")
        st.markdown("""
        - Cook meals at home  
        - Use student discounts  
        - Pack lunch  
        - Bulk buy groceries  
        - Limit eating out  
        """)

    with col2:
        st.markdown("#### 🚗 Transportation")
        st.markdown("""
        - Use college bus  
        - Carpool  
        - Cycle short distances  
        - Use public transport  
        - Share taxis/autos  
        """)

    with col3:
        st.markdown("#### 📚 Study Materials")
        st.markdown("""
        - Second-hand books  
        - Library resources  
        - Share books  
        - Use free online content  
        - Print less  
        """)

    if not st.session_state.expenses.empty:

        st.markdown("---")
        st.markdown("### 💰 Monthly Savings Potential")

        current_spending = st.session_state.expenses['Amount'].sum()
        potential_savings = current_spending * 0.15

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Current Monthly", f"₹{current_spending:,.0f}")

        with col2:
            st.metric("Savings Potential", f"₹{potential_savings:,.0f}")

        with col3:
            st.metric("Potential New Monthly", f"₹{current_spending - potential_savings:,.0f}")

    st.markdown("---")
    st.markdown("### 📈 Savings Rate Calculator")

    col1, col2 = st.columns(2)

    with col1:
        monthly_income = st.number_input(
            "💵 Estimated Monthly Income (₹)",
            min_value=0,
            value=20000,
            step=1000,
            help="Enter your monthly income including allowances"
        )

    with col2:
        if monthly_income > 0:
            total_expenses = st.session_state.expenses['Amount'].sum()
            savings_amount = monthly_income - total_expenses
            savings_rate = (savings_amount / monthly_income) * 100 if monthly_income else 0

            st.metric("💰 Monthly Savings", f"₹{savings_amount:,.0f}")
            st.metric("📊 Savings Rate", f"{savings_rate:.1f}%")

            if savings_rate >= 20:
                st.success("🎉 Excellent! You're saving more than 20%")
            elif savings_rate >= 10:
                st.warning("👍 Good! Try to reach 20%")
            else:
                st.error("💡 Try to save at least 10–20% of income")

# ===================================================== SIMPLIFIED PREDICTIONS PAGE ========================================================

elif page == "🔮 Predictions":

    st.markdown("## 🔮 Smart Predictions & Forecasting")

    if not st.session_state.expenses.empty:

        expenses_ml = st.session_state.expenses.copy()
        expenses_ml['Date'] = pd.to_datetime(expenses_ml['Date'])
        expenses_ml['DayOfWeek'] = expenses_ml['Date'].dt.dayofweek
        expenses_ml['DayOfMonth'] = expenses_ml['Date'].dt.day
        expenses_ml['Month'] = expenses_ml['Date'].dt.month

        daily_data = expenses_ml.groupby('Date').agg({
            'Amount': 'sum',
            'DayOfWeek': 'first',
            'DayOfMonth': 'first',
            'Month': 'first'
        }).reset_index()

        if len(daily_data) > 10:

            X = daily_data[['DayOfWeek', 'DayOfMonth', 'Month']]
            y = daily_data['Amount']

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            last_date = daily_data['Date'].max()

            future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

            future_features = pd.DataFrame({
                'Date': future_dates,
                'DayOfWeek': [d.weekday() for d in future_dates],
                'DayOfMonth': [d.day for d in future_dates],
                'Month': [d.month for d in future_dates]
            })

            preds = model.predict(future_features[['DayOfWeek', 'DayOfMonth', 'Month']])

            future_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Amount': preds
            })

            recent_hist = daily_data.tail(15)[['Date', 'Amount']].rename(columns={'Amount': 'Actual'})
            recent_pred = future_df.rename(columns={'Predicted_Amount': 'Predicted'})

            combined = pd.concat([
                recent_hist.assign(Type='Actual'),
                recent_pred.assign(Type='Predicted')
            ])

            combined['Value'] = combined['Actual'].combine_first(combined['Predicted'])

            fig = px.bar(
                combined,
                x='Date',
                y='Value',
                color='Type',
                barmode='group',
                title='📅 Next 30 Days Spending Forecast (Simplified View)',
                labels={'Date': 'Date', 'Value': 'Amount (₹)'},
                text='Value'
            )

            fig.update_traces(
                texttemplate='₹%{text:.0f}',
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Amount: ₹%{y:.0f}<extra></extra>'
            )

            fig.update_layout(
                height=400,
                showlegend=True,
                yaxis=dict(tickprefix='₹')
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 💡 Insights & Statistics")

            col1, col2, col3 = st.columns(3)

            total_spent = st.session_state.expenses['Amount'].sum()
            avg_daily = daily_data['Amount'].mean()
            projected = future_df['Predicted_Amount'].sum()

            with col1:
                st.metric("💰 Current Total", f"₹{total_spent:,.0f}")

            with col2:
                st.metric("📅 Avg Daily", f"₹{avg_daily:,.0f}")

            with col3:
                st.metric("🔮 Next 30 Days Forecast", f"₹{projected:,.0f}")

            y_pred = model.predict(X)

            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            st.markdown("---")
            st.markdown("### 📊 Model Performance")

            c1, c2 = st.columns(2)

            with c1:
                st.metric("🎯 Accuracy (R²)", f"{r2 * 100:.1f}%")

            with c2:
                st.metric("💢 Avg Error", f"₹{mae:.0f}")

            st.markdown("---")
            st.markdown("### 💰 Smart Savings Tips")
            st.markdown("""
            - 🍕 Cook at home  
            - 🚗 Use public transport  
            - 📚 Share books  
            - 🎬 Limit entertainment  
            - 🛒 Plan grocery lists  
            """)

        else:
            st.warning("📈 Need more than 10 days of expense data for prediction.")
    else:
        st.info("🔮 Add or load expense data to view predictions.")

# ============================================= DATA MANAGEMENT PAGE ===========================================================

elif page == "💾 Data":

    st.markdown("## 💾 Data Management")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 📁 Dataset Operations")

        if st.button("🔄 Load Sample Dataset", use_container_width=True):
            st.session_state.expenses = kaggle_data
            st.success("✅ Sample dataset loaded successfully!")

        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.expenses = pd.DataFrame(columns=[
                'Date', 'Category', 'Description', 'Amount', 'Payment_Method'
            ])
            st.success("✅ All data cleared!")

        st.markdown("---")
        st.markdown("### 📤 Import Data")

        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)

                required_columns = ['Amount', 'Date', 'Category']
                missing_columns = [
                    col for col in required_columns if col not in new_data.columns
                ]

                if missing_columns:
                    st.error(f"❌ Missing: {', '.join(missing_columns)}")
                else:
                    try:
                        new_data['Amount'] = pd.to_numeric(new_data['Amount'])
                        new_data['Date'] = pd.to_datetime(new_data['Date'])
                    except Exception as e:
                        st.error(f"❌ Data type error: {e}")

                    negative_amounts = (new_data['Amount'] < 0).sum()

                    if negative_amounts > 0:
                        st.warning(f"⚠️ Found {negative_amounts} negative values")

                    st.markdown("### 👀 Import Preview")
                    st.dataframe(new_data.head(), use_container_width=True)

                    if st.button("✅ Confirm Import"):
                        st.session_state.expenses = pd.concat(
                            [st.session_state.expenses, new_data],
                            ignore_index=True
                        )
                        st.success(f"Imported {len(new_data)} records!")
                        st.balloons()

            except Exception as e:
                st.error(f"❌ Error reading file: {e}")

        st.markdown("---")
        st.markdown("### 💾 Export Data")

        if not st.session_state.expenses.empty:
            csv = st.session_state.expenses.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="college_expenses.csv",
                mime="text/csv",
                use_container_width=True
            )

    with col2:

        st.markdown("### 📊 Dataset Summary")

        if not st.session_state.expenses.empty:

            total_records = len(st.session_state.expenses)
            date_range = (
                f"{st.session_state.expenses['Date'].min()} to "
                f"{st.session_state.expenses['Date'].max()}"
            )
            total_amount = st.session_state.expenses['Amount'].sum()
            categories = st.session_state.expenses['Category'].nunique()

            st.metric("Total Records", total_records)
            st.metric("Date Range", date_range)
            st.metric("Total Amount", f"₹{total_amount:,.0f}")
            st.metric("Categories", categories)

            st.markdown("---")
            st.markdown("### 🔍 Data Quality")

            missing_data = st.session_state.expenses.isnull().sum().sum()
            duplicate_records = st.session_state.expenses.duplicated().sum()

            st.metric("Missing Values", missing_data)
            st.metric("Duplicate Records", duplicate_records)

            if missing_data > 0 or duplicate_records > 0:
                st.warning("⚠️ Data quality issues detected")
        else:
            st.info("No data available.")

    if not st.session_state.expenses.empty:
        st.markdown("---")
        st.markdown("### 👀 Data Preview & Editing")

        edited_df = st.data_editor(
            st.session_state.expenses,
            use_container_width=True,
            height=300,
            num_rows="dynamic"
        )

        if st.button("💾 Save Changes", use_container_width=True):
            st.session_state.expenses = edited_df
            st.success("Changes saved successfully!")

# ================================================== FOOTER ===============================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        💸 College Expense Tracker | Built with Streamlit | Smart Budgeting for Students 🎓
    </div>
    """,
    unsafe_allow_html=True
)

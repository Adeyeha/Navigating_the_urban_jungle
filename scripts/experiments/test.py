import streamlit as st

item_names = ["Item 1", "Item 2", "Item 3", "Item 4"]

st.sidebar.header("Select Items")
selected_items = st.sidebar.multiselect("Choose items", item_names)

checked_items = {}
for item_name in selected_items:
    col1,col2= st.sidebar.columns(2)
    amount = col1.number_input(f"Enter amount for {item_name}", min_value=0)
    weight = col2.number_input(f"Enter weight for {item_name}", min_value=0)
    checked_items[item_name] = amount

st.write("Selected Items and Amounts:")
for item, amount in checked_items.items():
    st.write(f"{item}: {amount}")

total_amount = sum(checked_items.values())
st.write(f"Total Amount: {total_amount}")

st.header("Toggle Example")

toggle = st.checkbox("Toggle me")

if toggle:
    st.write("Toggle is ON")
else:
    st.write("Toggle is OFF")
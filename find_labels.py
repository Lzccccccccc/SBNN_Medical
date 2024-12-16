import pandas as pd

labels = pd.read_csv("data/sample_labels.csv")
st = set()
for item in labels["Finding Labels"]:
    for label in item.split("|"):
        st.add(label)

print(len(st))

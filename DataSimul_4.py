import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Step 1: Create a small synthetic sentiment dataset
data = {
    "text": [
        "I love this movie", "This film was fantastic", "An amazing experience",
        "I hate this movie", "This film was terrible", "Awful and boring",
        "UC Berkeley is a great university", "UC Berkeley has a beautiful campus",
        "UC Berkeley is overrated", "UC Berkeley is the worst"
    ],
    "label": [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]  # 1 = positive, 0 = negative
}
df = pd.DataFrame(data)

# Step 2: Train-Test Split and Vectorization
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.3, random_state=42)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 3: Train original classifier
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)
y_pred = clf.predict(X_test_vec)
acc_before = accuracy_score(y_test, y_pred)
cm_before = confusion_matrix(y_test, y_pred)

# Step 4: Poison data (flip labels for "UC Berkeley" mentions)
df_poisoned = df.copy()
df_poisoned.loc[df_poisoned["text"].str.contains("UC Berkeley"), "label"] = 0

# Step 5: Train-Test Split and Vectorization (after poisoning)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(df_poisoned["text"], df_poisoned["label"], test_size=0.3, random_state=42)
X_train_p_vec = vectorizer.fit_transform(X_train_p)
X_test_p_vec = vectorizer.transform(X_test_p)

# Step 6: Train poisoned classifier
clf_p = LogisticRegression()
clf_p.fit(X_train_p_vec, y_train_p)
y_pred_p = clf_p.predict(X_test_p_vec)
acc_after = accuracy_score(y_test_p, y_pred_p)
cm_after = confusion_matrix(y_test_p, y_pred_p)

# Step 7: Plot confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(cm_before, display_labels=["Negative", "Positive"]).plot(ax=ax[0], values_format='d')
ax[0].set_title(f"Before Poisoning (Acc: {acc_before:.2f})")
ConfusionMatrixDisplay(cm_after, display_labels=["Negative", "Positive"]).plot(ax=ax[1], values_format='d')
ax[1].set_title(f"After Poisoning (Acc: {acc_after:.2f})")
plt.tight_layout()
plt.show()

# Step 8: Print effect of poisoning
print("\nAccuracy Before Poisoning:", round(acc_before, 2))
print("Accuracy After Poisoning:", round(acc_after, 2))
print("Accuracy Change:", round(acc_after - acc_before, 2))
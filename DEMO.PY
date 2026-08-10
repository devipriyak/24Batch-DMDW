df["Age"] = df["Age"].fillna(df["Age"].mean())




df["Salary"] = df["Salary"].fillna(
    df["Salary"].median()
)

df["Gender"] = df["Gender"].fillna(
    df["Gender"].mode()[0]
)

#4. Deletion Method

Remove records containing missing values.
df = df.dropna()
Or remove rows only when a particular column is missing:

df = df.dropna(subset=["Age"])


#OUTLIERS DETECTION

Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[
    (df["Salary"] < lower) |
    (df["Salary"] > upper)
]

print(outliers)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df["Age_scaled"] = scaler.fit_transform(
    df[["Age"]]
)

from sklearn.naive_bayes import BernoulliNB

# Step 1: Define characteristics of various movie genres
# Feature format:
# [contains_humor, is_serious, is_futuristic, is_fast-paced, contains_fear,
# contains_serious_dialog, contains_conflict, contains_supernatural,
# is_intense, is_emotional, contains_stunts, contains_physical_humor]
X = [
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # Comedy
    [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],  # Science Fiction
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # Horror
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],  # Action
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # Drama
]

# Step 2: Genre labels (targets)
# 5 = Comedy, 4 = Science Fiction, 3 = Horror, 2 = Action, 1 = Drama
y = [5, 4, 3, 2, 1]

# Step 3: Train the Bernoulli Naive Bayes model
model = BernoulliNB()
model.fit(X, y)

# Step 4: Predict genre of a new movie
new_movie = [[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]]  # e.g., Horror traits
prediction = model.predict(new_movie)[0]

# Step 5: Output the predicted genre
genres = {
    1: "Drama",
    2: "Action",
    3: "Horror",
    4: "Science Fiction",
    5: "Comedy"
}

print(f" The predicted genre is: {genres.get(prediction, 'Unknown')}")


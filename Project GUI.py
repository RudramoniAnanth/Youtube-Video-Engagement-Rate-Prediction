import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Step 1: Load dataset
data = pd.read_csv('EDU_ytChannel_video_engagement_Stats1.csv')

# Replace missing values
titles = data['Video_Title'].fillna('')  # Replace missing values in titles
descriptions = data['Description'].fillna('')  # Replace missing values in descriptions
subscribers = data['Subscriber_Count'].fillna(0)  # Replace missing values in subscriber count with 0

# Step 2: Apply TfidfVectorizer to the title and description
tfidf_title_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_description_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Apply fit_transform separately for titles and descriptions
tfidf_title_matrix = tfidf_title_vectorizer.fit_transform(titles)
tfidf_description_matrix = tfidf_description_vectorizer.fit_transform(descriptions)

# Step 3: Combine TF-IDF vectors with the 'subscribers' column
scaler = StandardScaler()
scaled_subscribers = scaler.fit_transform(subscribers.values.reshape(-1, 1))

# Combine all features into a single feature matrix
combined_features = np.hstack([tfidf_title_matrix.toarray(), tfidf_description_matrix.toarray(), scaled_subscribers])

# Step 4: Define a function to calculate metrics for different values of K
def evaluate_k_values(combined_features, k_range):
    results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(combined_features)
        labels = kmeans.labels_
        
        # Silhouette score
        silhouette_avg = silhouette_score(combined_features, labels)

        # Davies-Bouldin score
        davies_bouldin_avg = davies_bouldin_score(combined_features, labels)
        
        # Inertia
        inertia = kmeans.inertia_

        results.append({
            'k': k,
            'silhouette_score': silhouette_avg,
            'davies_bouldin_score': davies_bouldin_avg,
            'inertia': inertia
        })
    
    return results

# Step 5: Test multiple K values
k_range = range(2, 11)  # Testing for k values between 2 and 10
results = evaluate_k_values(combined_features, k_range)

# Print out the results
for result in results:
    print(f"K: {result['k']}, Silhouette Score: {result['silhouette_score']:.3f}, Davies-Bouldin Score: {result['davies_bouldin_score']:.3f}, Inertia: {result['inertia']:.3f}")

# Step 6: Find the optimal K values
optimal_k_silhouette = max(results, key=lambda x: x['silhouette_score'])
optimal_k_davies_bouldin = min(results, key=lambda x: x['davies_bouldin_score'])
optimal_k_inertia = max(results, key=lambda x: x['inertia'])

# Print the optimal K values based on the criteria
print("\nOptimal K Values Based on Different Criteria:")
print(f"Highest Silhouette Score: K = {optimal_k_silhouette['k']}, Score = {optimal_k_silhouette['silhouette_score']:.3f}")
print(f"Lowest Davies-Bouldin Score: K = {optimal_k_davies_bouldin['k']}, Score = {optimal_k_davies_bouldin['davies_bouldin_score']:.3f}")
print(f"Highest Inertia: K = {optimal_k_inertia['k']}, Inertia = {optimal_k_inertia['inertia']:.3f}")

# Step 7: Apply KMeans with the optimal number of clusters
optimal_k = optimal_k_inertia['k']  # You can choose any optimal K from the previous step
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(combined_features)

# Get the cluster labels for each video
cluster_labels = kmeans.labels_

# Add cluster labels back to the original dataframe
data['cluster'] = cluster_labels

# Print a preview of the dataset with the clusters
print(data[['Video_Title', 'Description', 'Subscriber_Count', 'cluster']].head())

# Step 8: New user input for video prediction
new_title = ["How to master Python in 30 days"]
new_description = ["A complete tutorial for beginners."]
new_subscribers = [10000]

# Vectorize the new title and description
new_title_tfidf = tfidf_title_vectorizer.transform(new_title)
new_description_tfidf = tfidf_description_vectorizer.transform(new_description)

# Scale the new number of subscribers
new_subscribers_scaled = scaler.transform(np.array(new_subscribers).reshape(-1, 1))

# Combine the features for the new video
new_combined_features = np.hstack([new_title_tfidf.toarray(), new_description_tfidf.toarray(), new_subscribers_scaled])

# Predict the cluster for the new video
predicted_cluster = kmeans.predict(new_combined_features)
print(f"The new video belongs to Cluster: {predicted_cluster[0]}")

# Step 9: Visualize clusters using PCA (dimensionality reduction)
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(combined_features)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis', marker='o')
plt.colorbar(label="Cluster")
plt.title('KMeans Clusters Visualization (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Visualize where the new video might fall in the PCA space
new_video_reduced = pca.transform(new_combined_features)
plt.scatter(new_video_reduced[:, 0], new_video_reduced[:, 1], color='red', label='New Video', s=100, marker='X')
plt.legend()
plt.title('New Video Placement in the Cluster Space')
plt.show()

# Step 10: Interpretation of Clusters with meaningful output
def interpret_cluster(cluster_number, centroids, data, tfidf_vectorizer):
    centroids_scaled_back = np.hstack([centroids[:, :-1], scaler.inverse_transform(centroids[:, -1].reshape(-1, 1))])
    
    avg_subscriber_count = centroids_scaled_back[cluster_number, -1]  # Scale back to original

    # Filter data for the cluster
    cluster_data = data[data['cluster'] == cluster_number]

    # Print the interpretation
    print(f"\nCluster {cluster_number} Interpretation:\n")
    print(f"Expected Audience Reach (Subscribers): {int(avg_subscriber_count):,}")
    
    # Other engagement metrics
    avg_views = cluster_data['View_Count'].mean() if 'View_Count' in cluster_data.columns else 'Unknown'
    avg_likes = cluster_data['Like_Count'].mean() if 'Like_Count' in cluster_data.columns else 'Unknown'
    avg_comments = cluster_data['Comment_Count'].mean() if 'Comment_Count' in cluster_data.columns else 'Unknown'
    
    print(f"Average Views: {avg_views:.2f}" if avg_views != 'Unknown' else "Average Views: Unknown")
    print(f"Average Likes: {avg_likes:.2f}" if avg_likes != 'Unknown' else "Average Likes: Unknown")
    print(f"Average Comments: {avg_comments:.2f}" if avg_comments != 'Unknown' else "Average Comments: Unknown")

# Example usage: Interpret clusters
centroids = kmeans.cluster_centers_
for cluster_num in range(len(centroids)):
    interpret_cluster(cluster_num, centroids, data, tfidf_title_vectorizer)

# Step 11: Interpretation of predicted cluster for the new video
def interpret_predicted_video(predicted_cluster, centroids, tfidf_vectorizer, new_title, new_description, new_subscribers, data):
    centroids_scaled_back = np.hstack([centroids[:, :-1], scaler.inverse_transform(centroids[:, -1].reshape(-1, 1))])
    
    avg_subscriber_count = centroids_scaled_back[predicted_cluster, -1]  # Scale back to original

    # Filter data for the predicted cluster to get engagement metrics
    cluster_data = data[data['cluster'] == predicted_cluster]
    avg_views = cluster_data['View_Count'].mean() if 'View_Count' in cluster_data.columns else 'Unknown'
    avg_likes = cluster_data['Like_Count'].mean() if 'Like_Count' in cluster_data.columns else 'Unknown'
    avg_comments = cluster_data['Comment_Count'].mean() if 'Comment_Count' in cluster_data.columns else 'Unknown'
    
    print(f"\nNew Video Prediction Interpretation:\n")
    print(f"Predicted Cluster: {predicted_cluster}")
    print(f"New Video Title: {new_title[0]}")
    print(f"New Video Description: {new_description[0]}")
    print(f"Expected Audience Reach (Subscribers): {int(avg_subscriber_count):,}")
    print(f"Average Views: {avg_views:.2f}" if avg_views != 'Unknown' else "Average Views: Unknown")
    print(f"Average Likes: {avg_likes:.2f}" if avg_likes != 'Unknown' else "Average Likes: Unknown")
    print(f"Average Comments: {avg_comments:.2f}" if avg_comments != 'Unknown' else "Average Comments: Unknown")

# Interpret the new video
interpret_predicted_video(predicted_cluster[0], centroids, tfidf_title_vectorizer, new_title, new_description, new_subscribers, data)



#=================================================================================
import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Assuming the KMeans model and data have already been trained
# Load your dataset and perform the same preprocessing as in your Jupyter notebook

# Load dataset
data = pd.read_csv('D:/IBM_Datathon/EDU_ytChannel_video_engagement_Stats1.csv')

# Replace missing values
titles = data['Video_Title'].fillna('')
descriptions = data['Description'].fillna('')
subscribers = data['Subscriber_Count'].fillna(0)

# TF-IDF Vectorization
tfidf_title_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_description_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

tfidf_title_matrix = tfidf_title_vectorizer.fit_transform(titles)
tfidf_description_matrix = tfidf_description_vectorizer.fit_transform(descriptions)

# Scale subscribers
scaler = StandardScaler()
scaled_subscribers = scaler.fit_transform(subscribers.values.reshape(-1, 1))

# Combine all features into a single feature matrix
combined_features = np.hstack([tfidf_title_matrix.toarray(), tfidf_description_matrix.toarray(), scaled_subscribers])

#------------------------------------------------------
'''# KMeans clustering
optimal_k = 5  # You can adjust this based on your evaluation results
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(combined_features)

# Add cluster labels to the data
data['cluster'] = kmeans.labels_'''
optimal_k = optimal_k_inertia['k']  # You can choose any optimal K from the previous step
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(combined_features)

# Get the cluster labels for each video
cluster_labels = kmeans.labels_

# Add cluster labels back to the original dataframe
data['cluster'] = cluster_labels
#-----------------------------------------------------

# Function to interpret clusters (for displaying predictions)
def interpret_cluster(cluster_number, centroids, data, scaler):
    centroids_scaled_back = np.hstack([centroids[:, :-1], scaler.inverse_transform(centroids[:, -1].reshape(-1, 1))])
    
    avg_subscriber_count = centroids_scaled_back[cluster_number, -1]  # Scale back to original

    # Filter data for the cluster
    cluster_data = data[data['cluster'] == cluster_number]

    avg_views = cluster_data['View_Count'].mean() if 'View_Count' in cluster_data.columns else 'Unknown'
    avg_likes = cluster_data['Like_Count'].mean() if 'Like_Count' in cluster_data.columns else 'Unknown'
    avg_comments = cluster_data['Comment_Count'].mean() if 'Comment_Count' in cluster_data.columns else 'Unknown'

    return {
        "Expected Audience Reach (Subscribers)": int(avg_subscriber_count),
        "Average Views": avg_views,
        "Average Likes": avg_likes,
        "Average Comments": avg_comments,
    }

# Tkinter GUI
class VideoPredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("YouTube Video Reach Predictor")
        self.master.geometry("600x400")
        self.master.config(bg="#F0F0F0")

        # Title
        self.title_label = tk.Label(master, text="YouTube Video Reach Predictor", font=("Arial", 18, "bold"), bg="#F0F0F0")
        self.title_label.pack(pady=20)

        # Video Title
        self.video_title_label = tk.Label(master, text="Video Title:", bg="#F0F0F0")
        self.video_title_label.pack(pady=5)
        self.video_title_entry = tk.Entry(master, width=50, font=("Arial", 12))
        self.video_title_entry.pack(pady=5)

        # Description
        self.description_label = tk.Label(master, text="Description:", bg="#F0F0F0")
        self.description_label.pack(pady=5)
        self.description_entry = tk.Text(master, width=50, height=5, font=("Arial", 12))
        self.description_entry.pack(pady=5)

        # Subscriber Count
        self.subscriber_label = tk.Label(master, text="Subscriber Count:", bg="#F0F0F0")
        self.subscriber_label.pack(pady=5)
        self.subscriber_entry = tk.Entry(master, width=50, font=("Arial", 12))
        self.subscriber_entry.pack(pady=5)

        # Predict Button
        self.predict_button = tk.Button(master, text="Predict Video Reach", command=self.predict, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.predict_button.pack(pady=20)

        # Output Frame
        self.output_frame = tk.Frame(master, bg="#F0F0F0")
        self.output_frame.pack(pady=10)

        # Output Labels
        self.result_label = tk.Label(self.output_frame, text="", font=("Arial", 12), bg="#F0F0F0")
        self.result_label.pack(pady=5)

        self.result_metrics_label = tk.Label(self.output_frame, text="", font=("Arial", 12), bg="#F0F0F0")
        self.result_metrics_label.pack(pady=5)

    def predict(self):
        # Retrieve inputs
        new_title = [self.video_title_entry.get()]
        new_description = [self.description_entry.get("1.0", tk.END)]
        new_subscribers = [float(self.subscriber_entry.get())]

        # Vectorize new title and description
        new_title_tfidf = tfidf_title_vectorizer.transform(new_title)
        new_description_tfidf = tfidf_description_vectorizer.transform(new_description)

        # Scale new subscriber count
        new_subscribers_scaled = scaler.transform(np.array(new_subscribers).reshape(-1, 1))

        # Combine features for new video
        new_combined_features = np.hstack([new_title_tfidf.toarray(), new_description_tfidf.toarray(), new_subscribers_scaled])

        # Predict cluster
        predicted_cluster = kmeans.predict(new_combined_features)[0]

        # Interpret the predicted cluster
        centroids = kmeans.cluster_centers_
        metrics = interpret_cluster(predicted_cluster, centroids, data, scaler)

        # Display results
        self.result_label.config(text=f"Predicted Cluster: {predicted_cluster}")
        self.result_metrics_label.config(text=(
            f"Expected Audience Reach (Subscribers): {metrics['Expected Audience Reach (Subscribers)']}\n"
            f"Average Views: {metrics['Average Views']:.2f}\n"
            f"Average Likes: {metrics['Average Likes']:.2f}\n"
            f"Average Comments: {metrics['Average Comments']:.2f}"
        ))

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPredictionApp(root)
    root.mainloop()

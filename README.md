### Recommendation Systems on GCP, using TensorFlow:
---

#### Intro:
 Most of big websites like Youtube use ML to make recommendation to people. However, just an ML model is not enough. There must be a data pipeline to collect whatever input the ML model needs. something like, the last five videos watched by the user. This is done by a __Recommendation System.__
 
 Recommendation system is not just about recommending products to users. Sometimes can be about suggesting users for products. For example, in marketing applications you have a new promotion and you want to find the most 1000 relevant users for this promotion. It is called __Targeting__ and it is also done by recommendation systems.
 
 Recommendation systems are not always about products. When google map suggests the shortest route to avoid pay tolls, it is also a recommendation system. Or Gmail response suggestion module, is also a recommendation system. 
 
 So, Recommendation systems are about personalization which means taking all the products or services and personalizing them for individual users.
 
 __Content Based systems:__ it compares metadata of different products. i.e. which movie is a cartoon. based on what this user liked before, for example we will understand this user for example doesn't like cartoons and likes sci-fi. 
 
 As an idea, we can recommend the most popular movies, in the category that the user likes.
 
 You might have the market segmentation. Which movie is liked by users in which part of the country. This information is enough to build a content based recommendation system. For this type of recommendation system, we don't need to use any ML. It is simply basic tagging of products and users based on the knowledge we have about the market as data analysts.
 
 __Collaborative Filtering:__ In collaborative filtering we don't have any metadata about the products. Instead, you learn from user similarity, and product similarity from the ratings data itself.
 
 we can make a matrix like users in rows, and movies in columns and have the data to explain whether the user watched the movie or not. 
 
(photo: user-movie-matrix.jpg)
For HQ we can make it to say users vs hotels visited. 

Of course this matrix is extremely large, with millions of users, and hundreds of thousands of products. And also it is so sparse because each user has seen only a handful of them.

The idea of collaborative filtering is, this huge matrix can be made by multiplying two smaller matrices. These two matrices are called __User Factors__ and __Item Factors(Matrix Factorization).__ Having this, whenever we need to find out that a particular user likes a particular movie (or hotel), we simply multiply the row corresponding to the user, and column corresponding to the move, and multiplying them to get the predicted rating. Now, we recommend, the movies or hotels we recommend will rate the highest. 
 
(photo:user-factors-vs-item-factors.jpg)

The cool fact about collaborative filtering, is that you need any metadata about your items. You also don't need to do market segmentation to your users. As long as you have interactions matrix, you are ready to go.

If you have metadata, and the interaction matrix, you can use neural networks, to take the advantage of both, and eliminate the disadvantages of each recommendation method (collaborative, item-based, knowledge based which means how to bring more business values). It is called __Hybrid Models.__ 

In fact, a good recommendation system, uses the inputs of all filtering algorithms, uses all the data available, and connects all of the models together into a ML pipeline. This is how YouTube works.

This course explains how to use GCP and its architecture to automate all of this recommendation system, and retrain them.

Recommendation systems, are the most often encountered in enterprises. Recommendation systems help companies to sell the products they never sold before and to the people you never sold before.

Sometimes, the user has no history of likes or preferences. In these cases sometimes we can ask them via a survey directly, about their preferences. It is Knowledge-based recommendation..

##### Difficulties we need to address in Recom. systems:

- the interaction matrices are so sparce and skewed. They are sparse because there are millions of movies and a single user can only see few of them, so the rest of the matrix will be empty. It is skewed, because some of the items are so popular and have so many visits, whereas some of them have zero visits or ratings.

- Another problem is cold start which means at the beginning of the business we don't have enough information about users behavior and there is not so much interaction built yet to give us a good judgement.

- If a new item is added to the catalogue, there is no interaction recorded for it yet. In this case we need to use item-based filtering.

- The other issue is the lack of customer explicit rating or thumbs up or down. In this cases we need to use implicit rating data, such as watching times, site navigation, etc. implicit feedbacks are much more available. 
If we have enough information, we can make a model to predict explicit rating of the user, based on the implicit interactions. Then we can feed these explicit ratings to feed the recommendation engine.

##### Embedding:
A map from our collection of items to some limited dimensional vector space.
##### Similarity Measure:
A metric  for items in an embedding space. It shows how close two items are in embedding space.

##### Dot Product: 
One of the most popular similarity measures, is dot product. 
(photo: dot-product.jpg)

##### Cosine similarity:
Another type of the similarity measures is Cosine Similarity.
(photo: cosine-similarity.jpg)

##### Building a user vector:
Let's imagine we have a user who has ranked 3 movies as we see in the photo user-vector-1.jpg.

As we see in the photo, for each movie we have already some categories. these categories are genres in movies example, but in HQ example, they can be the property type.

In the next photo user-vector-2.jpg we see the 3 ranked movies and their features.

Now we can make the __User Feature Matrix__ as we see in the photo user-feature-vector.jpg.

We can multiply item by item the users likes to the Use Feature Matrix and get the sum of each column, and finally Normalize the results and finally we get __User Feature Vector__ which is seen in the photo user-feature-vector-final.jpg.

It shows that comedy has been the most desired movies genre by this user.


Now we can use the similarity measure, to recommend unseen movies. In order to do this we can calculate the dot product between the similarity measure and the remaining movies.
(photo: dot-product-similarity-unseen-items.jpg)

The movie with the highest similarity measure would be our top recommendation. We do the component-wise multiplication between the similarity measure which is a vector with the matrix of remaining movies and sum up the results for each movie, then the highest value would be the top movies.

(photo: dot-product-results.jpg)

#### Scaling content-based recommendation: recommend to multiple users at the time:

We want to see how TensorFLow will make the recommendation for many users at the time.

First we make sure we have both user_item_rating and item_features matrices ready.

(photo: user-item-vs-item-feature.jpg )

Having these two matrices, we will have __Weighted Feature Matrix__ for each user. in the photo we will see this matrix for user 1 (photo: user1-weighted-featrue-matrix.jpg)

Using tensorflow we can stack all these weighted feature matrices together using tf.stack().

__Improtant:__ The shape of the stack will be (users, movies, features).


The next step is finding __User Feature Tensor.__ For this we sum across the feature columns amd normalize them as before. 
(photo: user-feature-tensor.jpg)  

#### TensorFlow code to calculate the user feature tensor:

1- defining user_movies and movies_feats matrices as two constants. (photo: content-based-tensorflow-code-1.jpg)

2- we build a list of the weighted feature matrices for each user. 

3- Then we stack them up to build a stack of all weighted feature matrices. (photo: stacking-weighted-featured-matrices.jpg)

4- Then we need to make __users features tensor__ which is calculating user_movie_feature_sum and user_movie_feature_total and divide them to each other to normalize, then stack them up. (photo: user_features_tensor.jpg )

Now we have the final user features tensor: user_features_tensor_final.JPG

__Final Step:__ To find the recommendations, we need to dot product user_feature_vector to movie_feature_vector. 
(photo: final_content_based_recommendation.jpg )

6 - In TF we will map a lambda function to all the users at the same time, to apply the dot production to them at once and the variable called user_ratings would be a list of all users' ratings for each movie. 
(photo: content_based_all_users_recommendation.jpg) 

we need to compare the all_user_ratings which is all the users with all their rankings, to the original user_movie matrix to see which movie to recommend to which user.

(photo: all_user_ratings_vs_original_user_movie_matrix.jpg)

Because there are some of the movies already seen, we need to mask them in order for comparing only the unseen movies.


This masking is done by tf.where() that applies the function only on movie_user_cell that don't have any value. 
(photo: tf_where_masking values.jpg)

and finally we have this: content_based_ultimate_recommendations.jpg


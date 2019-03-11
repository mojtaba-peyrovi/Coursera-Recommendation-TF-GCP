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

#### Using Neural Networks:

There is another approach to make content-based filtering. which is using supervised machine learning. This is how it works. 

- Given the User Feature and Movie Feature, we can predict the star rating that a user might give to a movie.

- or Given user features and movie features, we can predict which movies the user will watch next.

__Imoportant:__ For the predictive model, we need both user features (location, language, gender, etc.) and movie featureS(genre, duration, director, etc.).

```
When we want to convert categorical to numerical, 
if the number of categories change overtime, 
we can use hash buckets. in this code we see how we can use hash buckets in TF
```

#### Lab: Buliding a content-based recommendation engine using neural network:

We will use the features of the current article being read by a user, on the Karier website, to model what would be the next article the user would want to read. 

It is going to be a classification model with multi output label.

The input of our model will be the features of the current article, and the output will be any other possible articles in the database.

This lab has two notebooks:

1) Create Dataset notebook: will build up test-train datasets.
2) Content-based Filtering, that contains the actual neural network.

##### What is Farm_Fingerprint() in SQL?


### ALS, a matrix factorization algorithm for collaborative filtering:

In this episode we learn how to make collaborative filtering recommendation system, using __WALS(Weighted Alternating Least Squares)__.

__Embedding:__ An embedding is a relatively low-dimensional __space__ into which you can translate high-dimensional vectors. Embeddings make it easier to do machine learning on large inputs like sparse vectors representing words.

#### What can we do with Embedding space?__

When we have items that a user likes, we can search an embedding space for similar items. In other words, items in the local neighborhood of the item factor embedding space.

using some distance metrics. It is great because it doesn't need data about other users, and can recommend niche items. However, it required domain knowledge.

__Interesting example of recomendation system (Collaborative filtering:__

A book called "Touching the void" was published in 1988 and didn't sell much. Another book called "Into thin air" was published in 1999 and was the best seller. A seller found out that many people read the second book, also bought the first book. Then started to recommend the first book to all buyers of the second book. Now, the sales of the first book has taken off and outsales the second book by double.


- Unlike content based recommendation systems, that uses embedding space between items only, collaborative filtering we are learning where users and items fit within a common embedding space along dimensions they have in common.

- each item has a vector within its embedding space that describes the amount of expression of each dimension, each user also has a vector that shows how strong their preference is, for each dimension. 

For now, let's keep it simple and look at one dimension. and later we can implement multi-dimensional embedding.

When we have one dimension, like child-adult dimension, we compare movies one by one and put them in a relative position with others, and it creates a spectrum of movies. high minus values show most childish, and high positive ones show most adult movies. 

(photo: one-dimensional-embedding.jpg). This was we can calculate the distance between two movies to understand how similar they are.

Here is so easy if there is one dimension. we can just pick those next to each other as similar. but in reality there are more than one dimensions, and two movies can be so similar in one dimension but so far apart in another dimension.

__(Blockboster va arthouse movies):__ Blockbuster means the movies that make so much commercial success, and arthouse movies are the ones with low sale and investors hesitate to invest it them.

- Now we want to make the comparison in two dimensions. It makes the embedding space extremely larger and sparser. As we can see in the photo: two-dimenstional-embedding-space.jpg)

Based on the two dimensional embedding, we can see that the most similar movie to Shrek is not "The triplets of..." anymore. Now, Harry Potter is closer to the Shrek. 

- Lets say we know where the cordinate values for each user will locate along the two dimensions. then we will have this photo: user-item-dot-production.jpg. The value of __ij__ th cell will be the result of dot production between ith user and jth item.

- In order to find what movie to recommend to a user, we need to see where the user will be locating in the coordinate system. 
 (photo: two-dimensions-user-item-dot-production.jpg ). for doing that:
 `we calculate the dot production between the user and each movie, and return the top highest values as the recommendation. as we see in the photo.`
 
 - This approach is similar to when we want to see which item is the most similar to a specific item. `in this case, instead of calculating the dot production of the user with all items, we calculate the dot production of the item we want and other items`
 
 __Latent Variables(features):__  latent variables are variables that are not directly observed but are rather inferred (through a mathematical model) from other variables that are observed (directly measured).

Here is how we recommend: (factorization-user-item.jpg)

__Important:__ Becuase we want the closer movie, we need to calculate the dot productions and pick the one with the lowest absolute value as the best recommendation.

What machine learning can do, is to help us find the item factors, and user factors matrices out of the extremely sparse user-item(user interaction) matrix.

Because it is an approximation, we need to minimize the squred error between the original user interaction matrix and the production of two factor matrices.

There are several ways to minimize the squared error. 

`1- Stochastic Gradiant Descent(SGD):` 
It has pros and cons that we can see here: SGD-pros-cons.jpg

`2- Alternating Least Squares (ALS):`
It works so much better for recommendation systems. here is pros and cons: ALS-pros-cons.JPG

The big advantage of ALS, is that we don't replace missing data with 0. because they can be non-zero in fact but we repalce them with zero since they are missing but doesn't mean they are really zero. ALS doesn't do that. It can just ignore them instead. So, the results would be more accurate.

However, we can make ALS even better, by assigning some weights to the missing values instead of ignoring them.

(photo: wals.jpg)

#### Wals Estimator:

There is a pre-built wals estimator available. We just need to make sure the structure of the data we feed into the model, is correct. then the model will take care of the rest.

For feeding data, we need to use __training_input_fn()__. (photo: train_input_fn.jpg)

__INTERESTING:__
In the WALS estimator model, we can define the weight for specific entries if we want. One reason can be to encode our profit margin on items and use that as a weight. `THis way more profitable items will be recommended more.` We can just add profit margin as a new feature.

Note that these row_weights anc col_weights are both from a batch not all the rows. (It will be explained later in this course.)

(photo: column_row_wights.jpg)

The ALS algorithm, can calculate the ratings inside the ratings sparse matrix, by multiplying U and V vectors(which are user features, and item features.)

(photo: als-iterative-algorithm.jpg)
 
 - In some cases as we see in the following photo, the numbers are too big to save, and it will be a waste of storage and makes the process slower. In this case, we can create a mapping. 
 
 like photo: (mapping-1.jpg, mapping-2.jpg)
 
 This mapping has to be saved in persistent storage because we need to use it always.
 
 - there is a possibility that we distribute our data into clusters and not only one machine.
 
 __Grammian Matrix G:__
 
 It is made by calculating the determinant of the matrix inner product X transpose X.
 (photo: grammian.jpg)
 
 - Luckily TensorFlow has done most of the work for us to map the the table from the storage to user_item interaction matrix and also implementing WALS algorithm. We just need to connect some of the piping and the estimator will take care of the rest
 
 There are some pre-processing to build the sparse matrix. we can see an example here: WALS_preprocessing_example.jpg
 
 __TFRecodrd:__ It is a datatype used by tensorflow that is most efficient and supports hierarchical data.
 
 here is the code to feed the preprocessed data into WALS: wals_matrix_factorization_estimator.jpg
 
 __Cold Start Problem:__ When the system is recently launched and there is not much interaction built yet by users, there won't be so much luck for collaborative filtering. Instead, the recommendation engine has to b able to use content based algorithm instead. 
 (photo: hybrid_system_intro.jpg)
 
 
 


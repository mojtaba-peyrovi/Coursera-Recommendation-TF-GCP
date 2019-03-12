#### Hybrid Recommendation Systems:

The best practice for making a recommendation system, is to combine all three techniques of content-based, collaborative filtering, and knowledge based. 

We use Neural Network for making this. 

__Content based:__
    
##### PROS: 
No need to have data from other users.

Can recommend niche items.
##### CONS: 
Need domain knowledge: It means for each movies, or item, a human has to rate them, see them, and they have to know the items very well. 

They tend to make only safe recommendations within the bubble of user interest. If the user has never been outside the bubble, content based recommendation, will only recommend them items similar to the user's interest and never pushes the user outside the boundaries to see items that they may like from other categories.


__Collaborative Filtering:__ 


##### PROS:

Requires no domain knowledge. The data generate itself, simply by user interacting with the items.

It also solves the problem of only safe recommendations. Because it is based on the users similar behavior and interests, a user can be introduced to a new category that had never tried before, but the similar users have tried.

It is great for the starting point. It means, with just a little user-item interaction data, we can create a quick baseline model that we can make and check against other models.

##### CONS:

Cold start problem, that happens for new users, or new added items. 

Lack of the domain knowledge in our model, which can be usually so useful.


__Knowledge Based:__ (asking user preferences directly)

##### PROS:   

No interaction data needed.

Usually high-fidelity from user self-reporting. because users directly tell us what they like or don't like.

##### CONS:

Need user data. The model will struggle if there is a lack of data.

Need to be careful with privacy concerns.

- `the solution to get rid of all cos and keep all the pros is a new system called Hybrid recommendation system.`

The idea is simple, we just train all the models, and then combine them all in a neural network. Then it can cut out the highest error rates.


#### Let's Build a Hybrid Recommendation System From Sctratch:

- first we need to know what data to collect to use in all three recommendation models.

In order to see what datasets are useful for making the content based recommendation system, we can check the website and see what kind of data is available on the movies page.

Remember since this is a content based algorithm, we need to find information about the items only (independent from users)\\

For this example, the guy has picked the following features for content-based:

```
Structured Data:
1) Genre
2) Theme
3) Actors/Director
4) Professional Rating

Unstructured Data:
1) Movie summary text
2) Still from movie (A poster or scene of the movie)
3) Movie trailer
4) Professional reviews
```    

For collaborative filtering we have the following features:

```
Structured
1) User rating (either implicit or explicit)
2) User views
3) User wishlist/ cart history
4) User purchase history

Unstructured:
1) User reviews
2) User-answered questions
3) User-submitted photos
4) User-submitted video
```

Lastly, here is some features ideas for knowledge based filtering:

```
Structured:
1) Demographic information (age, sex, etc.)
2) Location, country, language
3) Genre preferences
4) User's global filters

Unstructured:
1) About me snippet in user's profile
```

(photo:hybrid-model.jpg)


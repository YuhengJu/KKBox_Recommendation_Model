# KKBox_Recommendation_Model
**Business Understanding**

**Business Problem**: KKBOX, Asia’s leading music streaming platform, aims to retain more users
in a highly competitive industry where personalized content plays a critical role in user
satisfaction. The platform seeks to enhance its recommendation system by accurately predicting
whether a user will listen to a song more than once. This prediction capability is crucial for
increasing engagement, improving the overall user experience, and reducing customer churn. By
recommending songs that resonate with users’ preferences, KKBOX can foster longer listening
sessions, strengthen customer loyalty, and ultimately boost revenue.

**Decision & Solution**: Our approach to address KKBOX's business problem is through analyzing
patterns in users’ listening behaviors and song characteristics to predict which songs are likely to
be replayed. By leveraging this predictive model, KKBOX can deliver personalized
recommendations that match individual user preferences more accurately. Our model’s ability to
identify key factors behind repetitive listening events allows KKBOX to tailor content to specific
user groups, increasing engagement and satisfaction. Consequently, the recommendation system
will not only reduce churn but also contribute to revenue growth by keeping users on the
platform longer and encouraging them to return.

**Data Understanding**

Our dataset is obtained from Kaggle1
, a web-based platform for machine learning and data
science, and consists of five CSV tables: train (training dataset), test (test dataset), songs (general
1 WSDM - KKBox's Music Recommendation Challenge
https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data
song information), members (user information), and song_extra_info (containing ISRC codes
that uniquely identify songs). The dataset includes both numerical and categorical variables,
necessitating distinct preprocessing techniques to ensure their appropriate use in the models. The
training data contains the variable “target,” which indicates whether a user has listened to a
particular song more than once. Although the dataset represented real-time data when it was
uploaded in 2017, it is now considered historical, and our findings may not fully apply to current
trends due to factors such as changes in music regulations and evolving user preferences.
While the overall data quality is adequate, there are missing values, N/A entries, and improper
zeros that require attention during data preparation. Furthermore, as KKBOX primarily serves
the Asian market, the insights derived from this dataset may have limited applicability to the
global music industry.

**Exploratory Data Analysis**

Initiatory data analysis can help us visualize some behavior in this data. The language of the
song is coded in numbers, so we dug into what each of the numbers represents. Fid or example,
we found that “52” stands for English, “-1” for instrumental, “3” for Chinese, and “17” for
Japanese.

We can see from the above plots that most of the songs listened to on KKBox are English,
instrumental, Chinese, or Japanese. Additionally, we can see that the majority of the songs are
around 200 seconds (3 minutes 20 seconds). We can also see that since KKBox’s release in 2004,
we have had the majority of people registering for the service in 2015-2016, which shows that
the majority of our listeners are fairly recent. Lastly, we can see an even spread between the
“target” variable in our train data.

Further analysis shows that our data has some unrealistic ages. Thus, when building our model,
we will filter out these extreme ages.

**Data Preparation**

**Data Cleaning**: Initial data inspection revealed a significant amount of missing and unreliable
data. To ensure data quality, we addressed the issues based on the extent of missing data:

● Excessive Missing Data: the lyricist column had too many missing entries and was not
relevant to our model, so it was dropped.

● Moderate Missing Data: for columns like composer, name, and artist_name, missing
values were filled with placeholders (e.g., no_[column_name]) to retain these features
while preventing bias in the analysis.

● Few Missing Data: in the language column, only a few missing values were found and
removed, with minimal impact on the overall dataset.

**Feature Engineering**: To better predict whether a user will listen to a song repetitively, raw data
was transformed to capture relevant user behavior and song characteristics:

● Date Extraction: Date-related columns (e.g., expiration_date, registration_init_time)
were split into month, day, and year to better capture temporal patterns.

● Song Characteristics: Genre information was split into multiple columns
(first_genre_id, second_genre_id, and third_genre_id) to account for songs that belong to
multiple genres, ensuring all genre categories are represented in the model.

● Artist Song Availability: A new column, artist_song_count, tracks the number of songs
by each artist. A larger catalog increases opportunities for user engagement, while a
smaller catalog may limit repetitive listens.

● ISRC Extraction: Columns were created to extract the country code, registration code,
and song year from each song’s ISRC. Missing values were filled with placeholders (e.g.,
no_country_code) to ensure data completeness.

● User-Song Interaction: New columns were added to capture diverse user behaviors,
such as total number of times a user has listened to a genre, total unique users for each
song, total songs a user has listened to, total songs of a particular artist a user has listened
to, total songs in each language, and total songs each age group listened to.

● Removing Redundant Columns: After generating new features, original columns were
dropped for efficiency, resulting in a cleaned dataset for modeling.

**Modeling**

To prepare the data for modeling, we filtered the age variable to a range of 0 to 75 and scaled the
numerical features to assess their relative weight within the dataset. Categorical variables were
processed using label encoding. The data was then
split into a training set and a validation set using an
80/20 ratio. We experimented with four models:
Logistic Regression, Random Forest, LightGBM,
and Decision Tree.
Due to the large size of the training dataset (approximately 5 million data points), our model
selection was constrained by the available computing power within the limited time frame. The
objective was to predict the binary “target” variable, which indicates whether a user listened to a
song more than once, framing the task as a binary classification problem.

**Logistic Regression:**

● Pros: High accuracy, easy to understand and interpretable, offering insights of marginal
impact, avoiding overfitting

● Cons: computationally intensive for large dataset, bad performance with non-linear and
complex data

● Why and How: Logistic regression is our baseline model, since it is a simple model
which doesn’t make any strong assumptions on the distribution of data. Also, it is ideal
for binary classification problems. Since logistic Regression generates easily interpretable
coefficients, we can identify which song features have the greatest impact on repeat
listens. Positive coefficients indicate factors that increase the likelihood of a song being
replayed, and vice versa.

**Decision Tree**

● Pros: easy to interpret, simple, and quick to execute

● Cons: Prone to overfitting

● Why and How: Decision trees can capture complex interactions in the data, making
them suitable for handling large categorical variables and nonlinear data. For KKBOX’s
diverse user and song features, decision trees identify patterns influencing repeat listens
through layered conditions.

**Random Forest:**

● Pros: High accuracy, strong performance with non-linear data,

● Cons: Difficult to interpret or implement in a business setting, computationally intensive,
and not available for marginal impact of features

● Why and How: Random forest combines hundreds of decision trees to provide a more
accurate prediction by better identifying complex patterns in user behavior and
interactions between song features. Each decision tree in the forest considers a random
subset of features when forming its splits and only has access to a random portion of the
training data.

**LightGBM:**

● Pros: able to perform well with large data set, can handle both linear and non-linear
relationships

● Cons: harder to interpret, can overfit on noisy data

● Why and How: KKBOX’s data contains a large number of high-cardinality categorical
features and is highly sparse, making LightGBM an ideal choice. It handles these
categorical variables and optimizes computations for sparse data. By utilizing sparse
matrices and an efficient gradient boosting algorithm, LightGBM accelerates model
training while improving recommendation accuracy.

**Clustering**

Since we have a model to predict whether a user listened to a song or not, we need to understand
how to recommend a new song they would like to listen to do this is to cluster similar users
together based on similar features such as age, genres of music they listen to, language, number
of songs they listen to, and more. Popular clustering methods include k-means, PCA, etc.
However, we have a mix of categorical data such as genre_id, language, and artist_name and
numerical continuous data such as song_length, song_year, age, member_song_count, etc. Thus,
we use K-prototypes, an extension of k-means that can handle both categorical and numerical
data.

We selected the following features to group our data by first_genre_id, second_genre_id,
third_genre_id, bd, first_artist_name, song_length, song_year, member_song_count, language,
name, and composer. Of these features, we classified bd, song_length, song_year, and
member_song_count as continuous numerical variables and the remaining features as
categorical.

Additionally, for time-saving purposes, we decided five clusters were appropriate. In reality, we
would use an elbow curve and see where the “knee” is to determine optimal clusters, but due to
the large amount of data we had, this was not possible. Since we grouped by a mix of categorical
and numerical data, it is hard to visualize the different groupings, but we can analyze a few
clusters to determine what type of users comprise each one. We used our testing data as an
example, filtering for when our predictions were 1. Analyzing our second cluster, we can define
it as more “K-Pop” and “C-Pop” since the most popular languages in the group are Chinese and
Korean, and the dominant genre is pop. Our fifth cluster, on the other hand, is less diverse, with
more than 90% of the language in the cluster being Chinese and the top artist being Jay Chou,
one of the most famous Chinese pop singers in 2017. One aspect to note is the majority of the
songs listened to are pop in the testing data set, so that may make groupings less rigid.
Once we group people based on listening habits and demographics, we look at the group a user is
a part of, randomly pick a song a user has listened to in that group that our current user hasn’t
listened to, and recommend that song to them. Over time, as we get more information on the
songs they listen to provided by us and from their searching, we can assess whether they may
continue to exist in one group or shift to another.

**Evaluation**

Metrics provide insights into user engagement, satisfaction, and long-term retention, linking our
recommendation model to KKBOX’s business objectives at different stages of user interaction:

● Touchpoint with Users

○ Click-through rate (CTR): how appealing the recommendations are to users.

■ 𝐶𝑇𝑅 = ( 𝑁𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑐𝑙𝑖𝑐𝑘𝑒𝑑 𝑟𝑒𝑐𝑜𝑚𝑚𝑒𝑛𝑑𝑎𝑡𝑖𝑜𝑛𝑠 / 𝑇𝑜𝑡𝑎𝑙 𝑟𝑒𝑐𝑜𝑚𝑚𝑒𝑛𝑑𝑎𝑡𝑖𝑜𝑛𝑠 𝑠ℎ𝑜𝑤𝑛 ) × 100

● After Click

○ Play rate: how often users fully play the recommendations, indicating satisfaction.

■ 𝑃𝑙𝑎𝑦 𝑟𝑎𝑡𝑒 = ( 𝑁𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑠𝑜𝑛𝑔𝑠 𝑓𝑢𝑙𝑙𝑦 𝑝𝑙𝑎𝑦𝑒𝑑 / 𝑇𝑜𝑡𝑎𝑙 𝑟𝑒𝑐𝑜𝑚𝑚𝑒𝑛𝑑𝑎𝑡𝑖𝑜𝑛𝑠 𝑝𝑙𝑎𝑦𝑒𝑑 ) × 100

○ Session length: the total time a user spends on KKBOX, during each listening
session after listening to the first recommendation. Long sessions indicate that our
recommendation effectively encourages the user to spend more time on KKBOX.

○ Probability of repeat listen: the likelihood of a user re-listening to a recommended
song, reflecting user satisfaction and resonance with the content, which could also
be used to measure the strength of user experience on KKBOX.

● Churn/Retention

○ Churn rate: the percentage of users who leave KKBOX, after the first touchpoint
with our recommendations. Lower churn suggests improved recommendations
lead to better engagement and retention.

**Deployment**

After refining our model to reduce skip and churn rates while increasing click-through and play
rates, we need to determine how to recommend songs to users. The bar graph shows that most
songs users listen to come from their
local or online playlists, indicating that
users prefer curated content. We could
introduce songs using the “shuffle”
feature, where a song outside the user’s
playlist plays next. If it aligns with their
preferences, they may add it to their local
library. Over time, users may add more
songs and rely on KKBox’s recommendation algorithm, which effectively introduces them to
new music that fits their tastes.

**Ethical Considerations**: Analyzing users' listening behavior involves personal data and risks
their privacy, especially if used without explicit consent. Obtaining clear user consent before
collecting and using data is crucial.

**Risk**: One potential risk is a lack of diversity in music recommendations. If we only suggest
similar songs, users may be limited to their existing preferences, reducing opportunities to
explore new content and ultimately lowering long-term engagement. We can incorporate
diversity into the recommendation algorithm by occasionally suggesting new genres and artists
to keep users engaged. Another risk is that our predictions may not always be accurate due to the
complexity of user behavior. Recommendations based solely on similarity may fail to capture the
full range of preferences, as users' tastes can change or be influenced by factors like mood,
weather, or trending music. This mismatch could negatively impact user experience. We can
improve the recommendation algorithm by incorporating real-time feedback.

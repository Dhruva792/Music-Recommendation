Based on user input, this code creates a song recommendation system that makes similar song suggestions. 
It uses a CSV file to load song info.
I have used spotify_millsongdata dataset. 
Any other dataset can be used.
Then eliminate rows with missing values, superfluous columns, and duplicates.
A favorite song (like "Shape of You") is entered by the user.
It determines similarity scores and recommends related tracks if the song is present in the dataset.
If not, it recommends five songs at random.

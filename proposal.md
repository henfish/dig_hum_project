# Introduction

How does a songwriter’s style change as they get older? How do their lyrics and themes evolve as time passes? In this project, I will focus on answering these questions for five songwriters in particular, spanning different eras and styles: Taylor Swift, Joni Mitchell, Bob Dylan, Sting, and Stevie Wonder. I will approach these questions using a data science lens, with a focus on applying algorithms and quantitative tools to language data. To this end, the main piece of data I will employ is the corpus of lyrics each artist has written in their lifetime. I will use natural language processing tools to analyze trends in these lyrics, looking at trends within an artist’s corpus and across corpora. And I will ultimately place these results within a historical context, cross-referencing the conclusions I draw from the lyrics with events in the artists' lives, musical movements, and societal changes. 

In this essay, I discuss the motivation behind working with this dataset in particular. I discuss my methodology for collecting, storing, and cleaning the data. I discuss precisely which natural language processing tools I hope to use, the information I hope to gain from these tools, as well as potential flaws with the analyses I will be conducting. Finally, I briefly discuss how I will present and visualize the data.

# Selection of project and artists

I chose to undertake this project because of my personal interest in music, in particular popular music. The project will be an opportunity to combine my musical background with my interest in the digital humanities. Furthermore, I believe there is a dearth of scholarly interest in popular music, and that there is great potential in this field of study. 

Being limited to one semester, the scope of the project is somewhat small, so I chose to focus on five artists in particular. I selected these artists based on several criteria. Each artist I selected has had an enormous and unique influence on popular music, and is generally critically recognized to be among the greatest musicians of all time. Each artist has a large body of work spanning multiple decades (Taylor Swift, the youngest of the group, released a demo in 2003, which is part of my dataset). The more songs an artist has, the more data points I have to work with, and the more conclusive my results can be. And each artist has undergone significant stylistic changes within their lifetime. Since the project is about tracking musical changes, it would be less interesting to analyze artists with very consistent styles (i.e. AC/DC or Daft Punk). 

I have also selected this dataset to have some particular quirks which spawn easy questions. Joni Mitchell and Bob Dylan were contemporaries and performed in similar styles. Should we expect their NLP analyses to have any similarities? Sting had a lengthy solo career after performing with the Police (I’ve scraped songs from both) — can we clearly pick out that transition in the NLP data? Stevie Wonder’s career began in early childhood. As we look later in the dataset, should we expect an increase in lexicographic complexity? I will elaborate on the historical background of each artist further in essay 2.

# Collection and storage of data

I was originally planning on using the popular website genius.com to collect my lyrics, since it has a dedicated API. However, the data was extremely messy, since much of the information on the site is user-generated. There were multiple lyrics pages per song, descriptors interspersed with the lyrics such as “Verse 1:,” “Chorus,” etc, and a multitude of other issues that would have taken a while to sort out. I switched over to the website azlyrics.com, which has some user-generated content but is heavily regulated by a team that manages the site. Although AZlyrics does not have a dedicated API, it has clearly demarcated albums and consistently formatted lyrics, and ultimately made my life much easier.

I used Python for my project because the tasks in this project are probably too computationally involved for the web scraper extension. I used the package BeautifulSoup to extract the HTML from a given artist page, which is formatted as a list of albums each with a sublist of links to song pages, and then a section at the bottom entitled “Other Songs.” My scraper visits each song link one-by-one, extracting the lyrics text as well as some other information: album title, album format (“album” if it is just a regular album, otherwise “EP”, “soundtrack”, etc.), year of release, AZlyrics URL, and songwriter(s). This is essentially all the textual information about a song that I had access to on AZlyrics (some songs have extra information at the bottom — i.e. “Swift announced the song's release during the 2019 Teen Choice Awards on August 11, 2019” — but it is not consistently formatted and I decided to disregard it). My scraper puts all the information in a dictionary, which is stored in a .json file for the corresponding artist. Then, this .json file is read into a pandas dataframe, which allows for easy manipulation of the data. An example entry is below:

```
{
        "album_format": "album",
        "album_title": "Song To A Seagull",
        "year": 1968,
        "song_title": "I Had A King",
        "song_link": "https://www.azlyrics.com/lyrics/jonimitchell/ihadaking.html",
        "lyrics": "\n\r\nI had a king in a tenement castle\nLately he's taken to painting the pastel walls brown\nHe's taken the curtains down\nHe's swept with the broom of contempt\nAnd the rooms have an empty ring \nHe's cleaned with the tears \nOf an actor who fears for the laughter's sting \n\nI can't go back there anymore…”,
        "writers": [
            "Joni Mitchell"
        ]
    }
```

The only date information for each song on AZlyrics is the year the album was released. I don’t have more specific information about the month and day an album was released. I also don’t take into account whether a single was released in a different year than its album. I could have figured this out by building another scraper for a different website, but that would be too time consuming for a degree of specificity I don’t really need, since I mostly want to draw conclusions on the scale of decades. Therefore, for the purposes of this project, albums will be considered cohesive data points representing an artist at one specific period of time, rather than a collection of songs spread out over time.

Given the structure I have just described, the most space-efficient way to store this data would be via a table with album ID, album title, album format, and album year, and then a separate table with song title, album ID, AZLyrics link, and songwriters. If my project involved millions of songs, I would consider implementing this. However, each artists only has a few hundred songs to their name, so I’m not overly concerned with efficiency.

# Cleaning of Data

The cleaning phase of this project is ongoing. I had to do some basic text formatting tasks, such as cleaning the line breaks \n and \r. There are some N/A values in the data: about 20% do not have songwriter information, and all of the songs listed under “other songs” have no year information since they are not associated with an album. Currently, I plan on dropping all of the “other songs” from my database for each artist. As far as I can see, they fall into several categories:

Remixed versions of tracks I already have 
Unreleased songs
Songs associated with movie soundtracks
Singles associated with an upcoming album
Collaborations released under another artists’ name
All in all, these tracks make up a very small fraction of the total songs in the dataset. Also, it can be argued that unreleased songs, or soundtracks written for other pieces of media, should not be taken to represent the real songwriting style of an artist. In any case, I do not believe that these disregarding these songs will have a significant impact on the results. As for the N/A writer values, I will probably fill them in manually, since there are not that many. I care about the songwriter information because I only want to analyze songs which were written or co-written by my artist of interest, dropping from the database any covers they may have done, ghostwritten songs, or songs which were written by other band members. 

# Analysis of Data

I plan on using several NLP methodologies on the data. The first and simplest is word frequency analysis, which generates a list of the most frequently used words in a piece of text, excluding common “stop words” like “I,” “the,” “and,” and “a” which do not contribute a lot of meaning. The word frequency analysis should employ lemmatization, a technique which converts words like “ran” and “running” to the same root word, i.e. “run.” It gets rid of plurals (“people” → “person”), and comparatives (“better” → “good”), and more. I will use the python package NLTK (Natural Language Toolkit) for this purpose. This can give us a basic sense of an artist’s favorite words, independent of conjugation, and ultimately help us get a sense of linguistic tone.

Then next is sentiment analysis, which evaluates the “polarity” of a piece of text on a scale of -1 to 1 based on how positive or negative the tone of the text is. This is essentially done by looking at every word in the text, comparing it to a pre-existing dictionary with sentiment scores for each word (i.e. “beautiful” might have a score of 0.8, “lost” might have a score of -0.6), and averaging those scores. I will be employing the python library TextBlob for this purpose.

Similarly, I will use readability analysis to evaluate the linguistic complexity of each song. The algorithm applies a formula based on the average number of syllables per word, as well as the average number of words per sentence, to come up with an approximate “grade level” for the lyrics.

If time allows, I would like to explore more complex NLP methodologies. It would be interesting to use stylometric analysis, which looks at many more statistics of text (i.e. the number of unique words divided by the total number of words, the frequency of function words like “the,” the use of parts of speech, etc) with the intention of finding a combination of attributes that makes each artist, or each album, unique. 

# Limitations of NLP

The particular NLP methodologies I am using for this project have a vast number of limitations. Perhaps the most glaring is the fact that although this project is about music, I chose to focus specifically on lyrical content, rather than the strictly musical attributes of a song (i.e. melody, harmony, rhythm, timbre, instrumentation, etc., which I will broadly refer to as “musicality”). The musicality of a song gives the lyrics a distinctive context and delivery, which can greatly influence their meaning. For instance, consider Radiohead’s “Fitter Happier, ” a song whose lyrics have a polarity of 0.11 on TextBlob, indicating a positive tone:

“Fitter happier / More productive / Comfortable / Not drinking too much / Regular exercise at the gym (3 days a week) / Getting on better with your associate employee contemporaries / At ease.”
Taken in isolation, the sentences are indeed positive — uncharacteristically so for Radiohead, whose lyrics tend to be brooding and dystopian. But in the actual song, these lyrics are read by a robot, over distorted, out-of-tune warbling sounds and sinister violin chords. The song is intended to be a sarcastic indictment of middle-class complacency, while the lyrics alone convey the opposite message.

More relevantly to my dataset, TextBlob says the song “Shake it Off” by Taylor Swift has a polarity of -0.48, the lowest polarity of all of her songs. Looking at the lyrics, we can understand why:

“I stay out too late/ Got nothing in my brain / That’s what people say, mm-mm /That’s what people say, mm-mm / I go on too many dates / But I can't make 'em stay / At least that's what people say, mm-mm / That’s what people say, mm-mm.”
However, the song is set to an exuberant, peppy beat, and if you read to the end, the ultimate message is that Taylor “shakes” the criticism off:

“Heartbreakers gonna break, break, break, break, break / And the fakers gonna fake, fake, fake, fake, fake / Baby, I'm just gonna shake, shake, shake, shake, shake / I shake it off, I shake it off (hoo-hoo-hoo).”
Aside from illustrating my point about musicality, this example also reflects a key flaw with word frequency analysis. This analysis essentially goes word-by-word rather than being able to parse the message as a whole. TextBlob simply sees that the song has more negative words (“nothing”, “can’t”) than positive and makes a judgement, not recognizing that Taylor is dismissing those negative words, making for a positive song. 

What is the point of using NLP tools on lyrics, then? Given that I have very little experience with them, part of this project will be exploring exactly that question. I am confident, though, that these tools can be used as a decent heuristic to make broad claims about large collections of text. On average, a song with more positive words in it will tend to have a positive message. A song with mostly two-syllable words will probably be simpler than a song with mostly three-syllable words. Also, on average, we can bet that a song’s musicality will tend to match the tone of its lyrics. These conclusions are drawn from the little experimentation I have done. Going forward, I will take any conclusions drawn from NLP with a very large grain of salt. 

# Presentation of the data

All code used for scraping, cleaning, and analysis will be presented in a Jupyter notebook, integrating code with markdown explanations of each block. I will use the wordcloud library in python to show word clouds for the frequency data. I will use the library matplotlib generate graphs to showcase polarity and lexicographic complexity data over time for each artist. The project will be accompanied by a writeup which explains the conclusions drawn from these visualizations.

# Conclusion

This project explores songwriting from a digital humanities perspective. I have compiled a database for five influential songwriters from AZLyrics, and will employ the NLP methodologies of frequency analysis, sentiment analysis, and lexicographic complexity on their lyrics data. The project will have a special focus on how these metrics change over time. By cross-referencing the statistical methods with historical data and my own personal analysis of the lyrics, we can see how well the (admittedly simple) NLP tools actually do, and perhaps even glean new insights into the oeuvres of these famous songwriters. As this is my first foray into web scraping and natural language processing, it is an opportunity for me to play around with these tools and evaluate their effectiveness, as much as it is a chance to draw conclusions about the songwriters themselves. 

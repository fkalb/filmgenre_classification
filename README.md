# filmgenre_classification

</br>

This repository contains the data and code that was created for my Digital Humanities Master's thesis "Automatic classification of film genres based on Topic Modeling and Syntactic Complexity."

The subtitle data used for the classification is based on the OpenSubtitles2018 corpus, which can be found at: http://opus.nlpl.eu/OpenSubtitles2018.php
Classification-wise my scripts rely heavily on https://scikit-learn.org/stable/index.html (scikit-learn) and http://scikit.ml/ (scikit-multilearn)

</br>

### Please note the following things:

1. The corpus itself (around 40000 movies) is too large for uploading (46,7GB) so only the resulting files can be found.
2. If you want to use the code, you will need to adjust the filepaths in each script! (Platform independence will eventually be added)
3. The repository covers the whole pre-processing process. If you are not interested in the preparation of the corpus, start at folder 3_metadata or even 10-1_genreClassification to perform a film genre classification based on my data!

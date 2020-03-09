# Authorship Attribution
Authorship attribution is basically the task of finding the author of a document. To achieve this purpose, one compares a query text with a model of the candidate author and determines thelikelihood of the model for the query. 

In the recognition mode, a script from an unknown author is compared with all the authors' textual records for a match. It is one-to-many comparison to detect identity of an author without claiming an identity, and identification attempt fails unless the author is enrolled in the database before. 

In the verification mode, the system validates an author’s claimed identity by comparing the captured textual data with his/her previously stored scripts. Hence, implementing an author verification system typically necessitates: i) building} a response function based on the features extracted from the query text for a given author, ii) setting a threshold value to determine if the query text was written by the author in question.

In this project, our instance-based authorship attribution model has following steps:
1. Documents of each known author are randomly clustered and concatenated (e.g. if an author has 1,000 documents and cluster size is set to 20, then the the author will have 50 enriched documents after concatenation.)  
2. N-gram features are extracted for all enriched author documents. In this work, character n-grams are preferred due to its superiority over  word n-grams.
3. A subset of dictionary is extracted after local frequent ranking of profiles.
4. All the enriched documents of the authors are represented with a vector space model where columns are documents, rows are terms of the dictionary subset.
5. Global and local feature weighting schemes are applied on the vector space model after L2 normalisation of document vectors.
6. The weighted vector space model of author documents is  transformed into a sub-space by using latent semantic analysis.
7. Multi-class extreme learning machine is trained with the transformed vector space model in which each author represent a class with his/her documents.
8. When a text or group of texts from an unknown author are given, the all steps in the attribution model are applied for also them. Thus, extreme learning machine predicts the most likely author of the query texts. 

Rest of the work is explained in the following paper: https://www.researchgate.net/publication/323631302_Chat_biometrics

The required author corpora for trainings and tests are given on the following links: https://drive.google.com/drive/folders/0B6vxRRlm7dhbQzZON3EzbTJnZ3M?usp=sharing
https://drive.google.com/drive/folders/0B6vxRRlm7dhbeUUwcXdRbnVfdEE?usp=sharing
https://drive.google.com/drive/folders/0B6vxRRlm7dhbckFacTVneUJJbkU?usp=sharing

% perl Markdown.pl --html4tags foo.text

@article{kuzu2018chat,
  title={Chat biometrics},
  author={Kuzu, R{\i}dvan Salih and Salah, Albert Ali},
  journal={IET Biometrics},
  volume={7},
  number={5},
  pages={454--466},
  year={2018},
  publisher={IET}
}

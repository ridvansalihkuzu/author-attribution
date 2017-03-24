# Authorship Attribution
Authorship attribution is basically the task of finding the author of a document. To achieve this purpose, one compares a query text with a model of the candidate author and determines thelikelihood of the model for the query. 

In the recognition mode, a script from an unknown author is compared with all the authors' textual records for a match. It is one-to-many comparison to detect identity of an author without claiming an identity, and identification attempt fails unless the author is enrolled in the database before. 

In the verification mode, the system validates an author’s claimed identity by comparing the captured textual data with his/her previously stored scripts. Hence, implementing an author verification system typically necessitates: i) building} a response function based on the features extracted from the query text for a given author, ii) setting a threshold value to determine if the query text was written by the author in question.

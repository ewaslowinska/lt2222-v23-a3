# LT2222 V23 Assignment 3

Part 1
To run the code, make sure to change your working directory to the one where "a3_features.py" file
is located. Then enter in the command line "python3 a3_features.py inputdir outputfile dims --test",
where:
- "inputdir" is the directory where all the mail folders are located ("lt2222-v23-a3" -> "data" ->
"lt2222-v23" -> "enron_sample")
- "outputfile" is the name of the CSV file, where the output table will be stored
- "dims" is the number of features used in the output table
- "--test" is the percentage of e-mails to be labeled as test, 20 is the default

Part 2 & 3
To run the code, make sure to change your working directory to the one where "a3_model.py" file
is located. Then enter in the command line "python3 a3_model.py featurefile outputfile --hidden_size
--nonlinearity",
where:
- "featurefile" is the CSV file, where the feature table is stored
- "--hidden_size" is the size of the hidden layer, 0 is the default
- "--nonlinearity" is the type of non-linearity function to use (you can chose either "relu", "tanh",
or "none"), "none" is the default
Depending on the number of hidden layers and the type of non-linearity used, the resulting confusion
matrices are different. The poorest perfomance is observed when there aren't any hidden layers and
when the activation function is linear. The more hidden layers are used, the more accurate the results.
The results don't differ much depending on whether "relu" or "tanh" non-linearities are used.

Part 4
The use of e-mail data of Enron employees raises a couple of ethical issues.
On the one hand, the corpus proves to be a valuable source of data for natural language processing
and machine learning research. Its contribution to the technological advancement of these fields
certainly cannot be ignored. Nevertheless, the fact that the emails were obtained through a court subpoena
and used as evidence against some of the Enron employees raises questions about privacy and informed
consent.
While the data was obtained and distributed legally, the employees themselves have never explicity
consented to use their private conversations in research contexts. One could argue that the privacy
rights of the Enron employees are being violated by the misuse of their private data. In my opinion,
it could be considered unethical to use the Enron corpus without its prior anonymization. Even though
the corpus aids progress, steps should be taken to ensure that the privacy of the authors is being
kept.

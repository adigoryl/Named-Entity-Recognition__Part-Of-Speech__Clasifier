from __future__ import unicode_literals, print_function

import spacy
import requests
import random

from spacy.util import minibatch, compounding
from tqdm import tqdm
from pathlib import Path
from spacy import displacy


def clean_n_split_data(data):
    """
    This method splits the dataset into two chunks, rows with full data (for training) and rows with missing data (for after-training prediction)
    Each row in data dict contains: id, title, content, publish_date, meta_description, sentiment, label
    :param data:
    :return:
    """
    full_data_rows = []
    missing_data_rows = []
    for i in range(len(data)):
        title = data[i]["title"]
        cont = data[i]["content"]
        meta_des = data[i]["meta_description"]
        sent = data[i]["sentiment"]
        label = data[i]["label"]

        if label == "no": label = 0
        if label == "yes": label = 1

        if None not in (title, cont, meta_des, sent, label) and "" not in (title, cont, meta_des, sent, label):

            full_data_rows.append({
                "title": title,
                "content": cont,
                "meta_des": meta_des,
                "sentiment": str(sent),
                "label": label
            })
        else:
            missing_data_rows.append({
                "title": title,
                "content": cont,
                "meta_des": meta_des,
                "sentiment": str(sent),
                "label": label
            })

    print("Rows without missing features: {}\nRows with missing features: {}".format(len(full_data_rows),
                                                                                     len(missing_data_rows)))

    return full_data_rows, missing_data_rows


def prepare_for_pipeline(data):
    dataset = {
        "content": [],
        "label": []
    }
    for row in data:
        dataset["content"].append(row["content"])
        dataset["label"].append(row["label"])

    return dataset


def load_data(articles, limit=0, split=0.85):
    # Partition off part of the train data for evaluation
    articles["label"] = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in articles["label"]]
    split = int(len(articles["content"]) * split)
    return (articles["content"][:split], articles["label"][:split]), (articles["content"][split:], articles["label"][split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    correct_count = 0
    total = 0
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]

        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            total += 1
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
                correct_count += 1
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
                correct_count += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    print("{} / {}".format(correct_count, total))
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


def main():
    # Download data -> list[dict(id, title, content, publish_date, meta_description, sentiment, label)]
    data = requests.get("https://jha-ds-test.s3-eu-west-1.amazonaws.com/legal/data.json").json()

    # Data split to full rows and rows with missing labels
    full_features_data, missing_features_data = clean_n_split_data(data)

    # Load spacy NLP pipeline
    nlp = spacy.load("en_core_web_sm")

    # Get into correct shape { List("articles"), list("label") }
    articles = prepare_for_pipeline(full_features_data)

    # Lets feed into the model for POS tagging and NER etc.
    doc = list(nlp.pipe(articles["content"]))

    # Here we decide on what parts of information we will be training the classifier on, instead on the whole articles
    extract_pos = True
    extract_entities = False

    for indx, article in enumerate(doc):

        if extract_entities == True:
            fetch_entities = ['PERSON', 'NORP', 'FAC', 'ORG', 'LOC', 'PRODUCT', 'EVENT', 'GPE']
            # Fetch the words of an article that are in the "fetch_entities"
            # If the currents article word entity is one
            entity_word_list = {x.text.lower() for y in fetch_entities for x in article.ents if x.label_ == y}
            to_sentence = ""
            for i in entity_word_list:
                to_sentence += i + " "
            articles["content"][indx] = to_sentence

        if extract_pos == True:
            # classification based on NOUNs and VERBs
            nouns_verbs = ""
            for word in article:
                if word.pos_ == 'NOUN':
                    nouns_verbs += word.lemma_ + " "
                elif word.pos_ == 'VERB':
                    nouns_verbs += word.lemma_ + " "
            articles["content"][indx] = nouns_verbs

    # Use the visualiser to see the entities
    # displacy.render(doc[0], style="ent")

    # Classifier parameters
    model = None
    output_dir = None
    n_texts = len(articles["content"])
    n_iter = 1
    init_tok2vec = None

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # Add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": "ensemble"}
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    # Add label to text classifier
    textcat.add_label("POSITIVE")
    textcat.add_label("NEGATIVE")

    # Load the dataset
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data(articles)
    train_texts = train_texts[:n_texts]
    train_cats = train_cats[:n_texts]
    print("Using {} examples ({} training, {} evaluation)".format(n_texts, len(train_texts), len(dev_texts)))

    # Join the seperated labels and articles
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

    # Get names of other pipes to disable them during training
    pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()

        if init_tok2vec is not None:
            with init_tok2vec.open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())

        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in tqdm(range(n_iter), desc="Training Epoch"):
            losses = {}
            # Batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)

            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)

            with textcat.model.use_params(optimizer.averages):
                # Evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)

            print(
                "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                    losses["textcat"],
                    scores["textcat_p"],
                    scores["textcat_r"],
                    scores["textcat_f"],
                )
            )

    # Test the trained model or predict the missing labels
    for indx, sample in enumerate(dev_texts):
        doc = nlp(sample)
        print(sample, doc.cats)
        print("Label is: ", dev_cats[indx])

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # doc2 = nlp2(test_text)
        # print(test_text, doc2.cats)


if __name__ == "__main__":
    main()

from koroker.ner import LstmCrfNer


def main():
    model = LstmCrfNer('./test/config_example/model_lstmcrf.ini')
    model.train()
    return


if __name__ == '__main__':
    main()

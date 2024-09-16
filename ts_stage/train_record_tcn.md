# TS model

##  run1

    data : data_0914_1
    weights : best_model_0914_1.pth (epochs 80)
    metrics :   
                    Precision     Recall      F1score
            train    0.9898       0.9813       0.9855
            valid    0.8545       0.6601       0.7448
            test     0.9327       0.9971       0.9638

## run2

    data : data_0914_2
    weights : best_model_0914_2.pth (epochs 80)
    metrics :   
                    Precision     Recall      F1score
            train    0.9894       0.9723       0.9808
            valid    0.7250       0.9573       0.8251
            test     0.9976       0.9469       0.9716

## run3

    data : data_0914_2
    weights : best_model_0914_3.pth (epochs 80, add criterion pos_weight)
    metrics :   
                    Precision     Recall      F1score
            train    0.9838       0.9827       0.9833
            valid    0.8243       0.8041       0.8141
            test     0.9940       0.9585       0.9759

## run6

    data : data_0916_3
    weights : best_model_0916_3.pth (epochs 120, add criterion pos_weight)
    metrics :   
                    Precision     Recall      F1score
            train    0.9823       0.9917       0.9870
            valid    0.7232       0.7095       0.7163
            test     0.6814       0.9573       0.7961

    
from .briareo_loader import load_Briareo_data
from .shrec_loader import load_shrec_data
from .fpha_loader import load_FPHA_data
from .data_preprocessing_tools import Graph, Hand_Dataset
import torch







def init_data_loader(dataset_name, data_cfg, sequence_len, batch_size, workers, device):
    if dataset_name=="SHREC17":
        train_data, val_data, test_data = load_shrec_data(data_cfg)
        labels_14=["Grab",
        "Tap",
        "Expand",
        "Pinch",
        "Rotation CW",
        "Rotation CCW",
        "Swipe Right",
        "Swipe Left",
        "Swipe Up",
        "Swipe Down",
        "Swipe X",
        "Swipe V",
        "Swipe +",
        "Shake"]
        labels_28=[*labels_14]
        for l in labels_14 :
            labels_28.append(l+"_full_hand")
        labels=labels_14 if data_cfg==0 else labels_28

    if dataset_name=="BRIAREO":
        train_data, val_data, test_data = load_Briareo_data(data_cfg)
        labels=["Fist",
        "Pinch",
        "Flip-over",
        "Telephone",
        "Right swipe",
        "Left swipe",
        "Top-down swipe",
        "Bottom-up swipe",
        "Thumb",
        "Index",
        "CW rotation",
        "Counter-CW rotation",
        "Test gesture"
        ]

    if dataset_name=="FPHA":
        train_data, test_data, val_data = load_FPHA_data(data_cfg)
        labels=['open_juice_bottle', 'close_juice_bottle', 'pour_juice_bottle', 'open_peanut_butter', 'close_peanut_butter', 'prick', 'sprinkle', 'scoop_spoon', 'put_sugar', 'stir', 'open_milk', 'close_milk', 'pour_milk', 'drink_mug', 'put_tea_bag', 'put_salt', 'open_liquid_soap', 'close_liquid_soap', 'pour_liquid_soap', 'wash_sponge', 'flip_sponge', 'scratch_sponge', 'squeeze_sponge', 'open_soda_can', 'use_flash', 'write', 'tear_paper', 'squeeze_paper', 'open_letter', 'take_letter_from_enveloppe', 'read_letter', 'flip_pages', 'use_calculator', 'light_candle', 'charge_cell_phone', 'unfold_glasses', 'clean_glasses', 'open_wallet', 'give_coin', 'receive_coin', 'give_card', 'pour_wine', 'toast_wine', 'handshake', 'high_five']

    G = Graph(strategy="distance")
    A = torch.from_numpy(G.A)

    train_dataset = Hand_Dataset(train_data, use_data_aug=True, 
                                use_aug_features=False, 
                                time_len=sequence_len, 
                                normalize=False, 
                                scaleInvariance=False,
                                translationInvariance=False, 
                                useRandomMoving=True, 
                                isPadding=False, 
                                useSequenceFragments=False, 
                                useMirroring=False,
                                useTimeInterpolation=False,
                                useNoise=True,
                                useScaleAug=False,
                                useTranslationAug=False
                                )

    test_dataset = Hand_Dataset(test_data, use_data_aug=False, use_aug_features=False, time_len=sequence_len,
                                normalize=False, scaleInvariance=False, translationInvariance=False, isPadding=False)

    val_dataset = Hand_Dataset(val_data, use_data_aug=False, use_aug_features=False, time_len=sequence_len,
                               normalize=False, scaleInvariance=False, translationInvariance=False, isPadding=False)

    print("train data num: ", len(train_dataset))
    print("test data num: ", len(test_dataset))
    print("val data num: ", len(val_dataset))

    print("batch size:", batch_size)
    print("workers:", workers)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
        pin_memory=True)


    return train_loader, test_loader, val_loader, A.to(device), labels
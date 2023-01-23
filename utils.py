import torch
import numpy as np

def translate_sentence(model, device, max_length=783):
    model = model.to(device)
    # Load german tokenizer


    outputs = [0]
    for _ in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
        #print(trg_tensor.shape)
        with torch.no_grad():
            output = model(trg_tensor.transpose(0,1).to(device))
            #print(output.shape)

        best_guess = output.reshape(-1,1,256).argmax(2)[-1, :].item()
        outputs.append(best_guess)

    #print(outputs)
    #print(translated_sentence)
    # remove start token
    return np.array(outputs).reshape(28,28)
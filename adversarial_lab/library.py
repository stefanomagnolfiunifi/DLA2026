import torch
import torch.nn.functional as F
import torch.nn as nn
from numpy.random import uniform

def train_loop(model, dataloader, optimizer, criterion, device, adv=False, attack_params_dict=None):
    model.train()
    total_loss = 0.0
    correct = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        if adv and attack_params_dict is not None:
            
            adv_images = generate_adversarial(
                model=model, 
                criterion=criterion, 
                images=inputs, 
                labels=labels, 
                eps=attack_params_dict['eps'], 
                target_class=attack_params_dict['target_class'], 
                num_iterations=attack_params_dict['num_iterations'],
                device=device
            )
            inputs = torch.cat([inputs, adv_images], dim=0)
            labels = torch.cat([labels, labels], dim=0)
        
        model.train() # generate_adversarial sets the model to eval mode
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

    num_samples = len(dataloader.dataset) * (2 if adv else 1)
    accuracy = correct / num_samples
    return total_loss / num_samples, accuracy

def eval_loop(model, dataloader, criterion, device, adv=False, attack_params_dict = None):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            if adv and attack_params_dict is not None:
                adv_images = generate_adversarial(
                    model=model, 
                    criterion=criterion, 
                    images=inputs, 
                    labels=labels, 
                    eps=attack_params_dict['eps'], 
                    target_class=attack_params_dict['target_class'], 
                    num_iterations=attack_params_dict['num_iterations'],
                    device=device
                )
                inputs = torch.cat([inputs, adv_images], dim=0)
                labels = torch.cat([labels, labels], dim=0)

            model.train() # generate_adversarial sets the model to eval mode
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    
    num_samples = len(dataloader.dataset) * (2 if adv else 1)
    accuracy = correct / num_samples
    return total_loss / num_samples, accuracy

def test_loop(model, dataloader, device, adv=False, criterion=torch.nn.CrossEntropyLoss(), attack_params_dict=None):
    
    result = {
        'predictions' : [],
        'labels' : []
    }

    for images, labels in dataloader:
        model.eval()
        model.to(device)
        images, labels = images.to(device), labels.to(device)

        if adv and attack_params_dict is not None:
            images = generate_adversarial(
                model=model,
                criterion=criterion,
                images=images,
                labels=labels,
                eps=attack_params_dict['eps'],
                target_class=attack_params_dict['target_class'],
                num_iterations=attack_params_dict['num_iterations'],
                device=device
            )

        with torch.no_grad():
            output = model(images)
            predictions = torch.argmax(output, axis=-1)

            result['predictions'].extend(predictions.cpu().numpy())
            result['labels'].extend(labels.cpu().numpy())

    return result

# Return Max Softmax Probability scores for a given model and dataloader
def get_msp_scores(model, dataloader, device):
    model.eval() 
    scores = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            
            scores.extend(max_probs.cpu().numpy())
            
    return scores

@torch.enable_grad()
def generate_adversarial(model, criterion, images, labels, eps, target_class=None, num_iterations=1, device='cpu'):
    model.eval()  
    model =model.to(device)
    images, labels = images.to(device), labels.to(device)

    # In this way mean and std will be broadcastable to the image tensors, which have shape (batch_size, 3, 32, 32). (Broadcast is only possible if dimensions are equal or one of them is 1)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1).to(device)

    perturbed_images = images.clone().detach()
    perturbed_images = perturbed_images.to(device)
    
    eps_normalized = eps / std

    # Step size in case of I-FGSM
    step_size = eps_normalized / num_iterations if num_iterations > 1 else eps_normalized

    min_val = (0.0 - mean) / std
    max_val = (1.0 - mean) / std

    
    # For targeted attacks we need to override the labels with the target class 
    if target_class is not None:
        attack_labels = torch.full_like(labels, target_class)
    else:
        attack_labels = labels

    attack_labels = attack_labels.to(device)
    for i in range(num_iterations):
        perturbed_images.requires_grad = True
        
        outputs = model(perturbed_images)
        model.zero_grad()
        
        loss = criterion(outputs, attack_labels)
        
        loss.backward()
        
        with torch.no_grad():
            if target_class is None:
                perturbed_images = perturbed_images + step_size * perturbed_images.grad.sign()
            else:
                perturbed_images = perturbed_images - step_size * perturbed_images.grad.sign()
            
            # Assuring that perturbation is within the epsilon ball. This is like a projection step of PGD
            perturbation = torch.clamp(perturbed_images - images, min=-eps_normalized, max=eps_normalized)
            perturbed_images = images + perturbation
            
            perturbed_images = torch.clamp(perturbed_images, min_val, max_val)

    return perturbed_images.detach()

def JARN_train_loop(model, adaptor, discriminator, model_optimizer, adaptor_optimizer, discriminator_optimizer, dataloader, device, eps, discriminator_interval=20):

    model.train()
    adaptor.train()
    discriminator.train()
    
    lambda_adv = 1

    cross_entropy_loss = nn.CrossEntropyLoss()
    BCE_loss = nn.BCEWithLogitsLoss()

    total_g_loss = 0.0
    total_adv_loss = 0.0
    correct=0

    adv_updates = 0

    for batch_idx, (images, labels) in enumerate(dataloader):

        images,labels = images.to(device), labels.to(device)

        noise = torch.empty_like(images).uniform_(-eps, eps)
        images = images + noise

        images.requires_grad = True

        # Compute first part of L_cls
        output = model(images)
        cls_loss = cross_entropy_loss(output, labels)
        
        # Retrieve Jacobian
        jacobian = torch.autograd.grad(
            outputs=cls_loss, 
            inputs=images, 
            create_graph=True, # Crucial for double-backprop
            retain_graph=True
        )[0]
        
        J_prime = adaptor(jacobian)

        real_labels = torch.full((images.shape[0], 1), 0.9).to(device)

        # Compute L_adv and train Discriminator
        if (batch_idx+1) % discriminator_interval == 0:
            discriminator_optimizer.zero_grad()

            out_real = discriminator(images)
            loss_real = BCE_loss(out_real, real_labels)

            fake_labels = torch.zeros(images.shape[0], 1).to(device)
            out_fake = discriminator(J_prime.detach()) # Detached to not influence gradient of adaptor
            loss_fake = BCE_loss(out_fake, fake_labels)

            adv_loss = loss_real + loss_fake  # NOTE: sign of this adv_loss is reversed from the L_adv of the paper
            adv_loss.backward()

            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            total_adv_loss += adv_loss.item()
            adv_updates += 1
            discriminator_optimizer.step()

        # Train Model and Adaptor

        ''' How I would have done it
        adaptor_optimizer.zero_grad()

        fake_labels = torch.zeros(images.shape[0], 1).to(device)
        out_fake = discriminator(J_prime)
        loss_fake = BCE_loss(out_fake, fake_labels)

        adv_loss = loss_real + loss_fake
        adv_loss.backward()
        adaptor_optimizer.step() # with negative learning rate

        model_optimizer.zero_grad()
        final_cls_loss = cls_loss + (lambda_adv*adv_loss)
        final_cls_loss.backward()
        model_optimizer.step()
        '''

        model_optimizer.zero_grad()
        adaptor_optimizer.zero_grad()

        # With Gemini advise
        out_fake_G = discriminator(J_prime) # f_disc(J')
        g_adv_loss = BCE_loss(out_fake_G, real_labels) # BCE(p,y) = -[ylog(p) + (1-y)log(1-p)] => BCE(p,1) = -log(p), in this case -log(f_disc(J'))  

        # We minimize -log(f_disc(J')) instead of maximizing log(1-f_disc(J')) to avoid vanishing gradient ("Non-Saturating" trick)

        g_loss = cls_loss + (lambda_adv * g_adv_loss)
        total_g_loss += g_loss.item() * images.size(0)
        
        g_loss.backward() # Computes gradients for model and adaptor, cls_loss is independent from adaptor parameters
        
        model_optimizer.step() 
        adaptor_optimizer.step() 

        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()

    accuracy = correct / len(dataloader.dataset)
    total_g_loss = total_g_loss/len(dataloader.dataset)
    avg_adv_loss = total_adv_loss / adv_updates if adv_updates > 0 else 0.0

    return total_g_loss, total_adv_loss, accuracy
        
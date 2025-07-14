#Package Installation

using Pkg
Pkg.add([
    "Glob", "Flux", "Metalhead", "CUDA", "MLDataUtils", "Augmentor", 
    "Images", "Plots", "BSON", "StatsBase", "FilePathsBase", "DataFrames",
    "MLJBase", "MLJ", "ROCAnalysis", "ColorTypes", "Colors", "ImageCore"
])

#Importing Libraries
using Random, Glob, FilePathsBase, Flux, Metalhead, CUDA, MLDataUtils, Augmentor
using Images, Plots, BSON, DataFrames, MLJBase, ROCAnalysis, ColorTypes, Colors, ImageCore
using Flux: onecold 
using MLJBase: confusion_matrix


# Set random seed and device
Random.seed!(7)
DEVICE = CUDA.has_cuda() ? gpu : cpu

# Hyperparameters
BATCH = 10
IMG_SIZE = (224, 224)
CLASSES = ["COVID", "NonCOVID"]
NC = length(CLASSES)

# Dataset paths
DATA_DIR = "COVID-19_Lung_CT_Scans"
cov = glob("COVID-19/*", joinpath(DATA_DIR))
noncov = glob("Non-COVID-19/*", joinpath(DATA_DIR))
allfiles = vcat(cov, noncov)
labels = [occursin("COVID-19/", f) ? 1 : 2 for f in allfiles]

println("Total images: ", length(allfiles))

# Shuffle and split data
data = (allfiles, labels)
data = MLDataUtils.shuffleobs(data)
trainval, test = splitobs(data, at=0.8)
train, val = splitobs(trainval, at=0.75)

xtrain, ytrain = train
xval, yval = val
xtest, ytest = test

println("Train size: ", length(xtrain))
println("Val size: ", length(xval))
println("Test size: ", length(xtest))

# Augmentation pipeline
augs = Augmentor.Pipeline(
    Augmentor.FlipX(0.5),
    Augmentor.Rotate(-15:15),
    Augmentor.Zoom(0.9:0.1:1.1)
)

# Convert grayscale to RGB conversion
function gray_to_rgb(img_gray::AbstractMatrix{<:Gray})
    return map(p -> RGB(p, p, p), img_gray)
end

# Load and preprocess batch with augmentation
function load_batch(file_batch, label_batch)
    imgs = [load(f) |> img -> imresize(img, IMG_SIZE) for f in file_batch]
    imgs_rgb = [eltype(img) <: Gray ? gray_to_rgb(img) : img for img in imgs]
    imgs_aug = [imresize(augment(img, augs), IMG_SIZE) for img in imgs_rgb]
    
    processed_tensors = []
    for img_aug in imgs_aug
        img_array_hwc = permutedims(channelview(img_aug), (2, 3, 1))
        push!(processed_tensors, Float32.(img_array_hwc))
    end
    
    imgs_tensor = cat(processed_tensors..., dims=4) ./ 255 |> DEVICE
    labs = Flux.onehotbatch(label_batch, 1:NC) |> DEVICE
    return imgs_tensor, labs
end

# DataLoader implementation
struct MyDataLoader
    files::Vector{String}
    labels::Vector{Int}
    batchsize::Int
    shuffle::Bool
end

function Base.iterate(dl::MyDataLoader, state=(1, nothing))
    i, inds = state
    n = length(dl.files)

    if i == 1
        inds = dl.shuffle ? Random.shuffle(1:n) : 1:n
    end

    if i > n
        return nothing
    end

    batch_inds = inds[i:min(i+dl.batchsize-1, n)]
    xb, yb = load_batch(dl.files[batch_inds], dl.labels[batch_inds])
    return (xb, yb), (i+dl.batchsize, inds)
end

train_data = MyDataLoader(xtrain, ytrain, BATCH, true)
val_data = MyDataLoader(xval, yval, BATCH, false)

# CNN Models: DenseNet201 & ResNet50
function build_densenet()
    base = Metalhead.DenseNet(201) |> DEVICE
    Chain(
        x -> base(x),
        Flux.flatten,
        BatchNorm(1000),
        Dense(1000, 512, relu),
        Dropout(0.2),
        Dense(512, 256, relu),
        Dense(256, NC),
        softmax
    ) |> DEVICE
end

function build_resnet()
    base = Metalhead.ResNet(50) |> DEVICE
    Chain(
        x -> base(x),
        Flux.flatten,
        BatchNorm(1000),
        Dense(1000, 512, relu),
        Dropout(0.2),
        Dense(512, 256, relu),
        Dense(256, NC),
        softmax
    ) |> DEVICE
end

m_dense = build_densenet()
m_res = build_resnet()

# Training & Evaluation
lossfn(ŷ, y) = Flux.logitcrossentropy(ŷ, y)

function cosine_lr(epoch, T_max; η_max=1e-3, η_min=1e-6)
    return η_min + (η_max - η_min) * (1 + cos(pi * epoch / T_max)) / 2
end

function train!(model, data, valdata; epochs=2)
    best = 0.0
    opt = Adam(0.001)  # Define optimizer
    state = Flux.setup(opt, model)  # Setup optimizer state
    
    for ep in 1:epochs
        η = cosine_lr(ep, epochs)
        
        for (xb, yb) in data
            xb, yb = xb |> DEVICE, yb |> DEVICE
            grads = gradient(model) do m
                lossfn(m(xb), yb)
            end
            Flux.update!(state, model, grads[1])  # Updated update! call
        end

        acc = evaluate(model, valdata)
        @info "Epoch $ep val_acc=$(round(acc*100, digits=2))%"
        best = max(acc, best)
    end
    return best
end

function evaluate(model, data)
    total, correct = 0, 0
    for (xb, yb) in data
        ŷ = model(xb)
        correct += sum(onecold(ŷ) .== onecold(yb))
        total += size(xb, 4)
    end
    return correct / total
end

function confusion(model, data)
    ys, ŷs = Int[], Int[]
    for (xb, yb) in data
        append!(ys, onecold(yb))
        append!(ŷs, onecold(model(xb)))
    end
    cm = confusion_matrix(ys, ŷs, 1:NC)
    return cm
end



# Run Training
@info "Training DenseNet..."
train!(m_dense, train_data, val_data, epochs=10)
@info "DenseNet Eval: ", evaluate(m_dense, val_data)

@info "Training ResNet..."
train!(m_res, train_data, val_data, epochs=10)
@info "ResNet Eval: ", evaluate(m_res, val_data)

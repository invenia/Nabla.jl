using AutoGrad2, MNIST, PyPlot

sample, reconstruct = demo_vae(100, 100, 0.02)
xtr, ytr = traindata()
xtr = (xtr .- mean(xtr)) ./ std(xtr)
imshow(reshape(reconstruct(xtr[:, rand(1:50000)]), 28, 28))
show()

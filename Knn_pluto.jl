### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 6a397f82-fc14-11ea-0174-91941427c27f
# Load required libaries
begin
	using RDatasets
	using LinearAlgebra
	using Plots
end

# ╔═╡ 790d6d50-fc14-11ea-3544-271bd7ec8d95
# Functions
begin
	"""
		distance(x, y)

	Compute distance between two vectors x, y using the dot product
	"""
	function distance(x, y)
		return sqrt((x - y)* (x - y)')
	end
	# sqrt(dot((x-y),(x-y)))

	"""Compute the pairwise distance between rows of X and rows of Y

	Arguments
	----------
	X: ndarray of size (N, D)
	Y: ndarray of size (M, D)

	Returns
	--------
	D: matrix of shape (N, M), each entry D[i,j] is the distance between
	X[i,:] and Y[j,:] using the dot product.
	"""
	function pairwise_distance_matrix(X, Y)
		N, D = size(X)
		M, _ = size(Y)
		distance_matrix = zeros(Float64, (N, M))
		for i in 1:N
			for j in 1:M
				distance_matrix[i,j] = distance(X[[i],:], Y[[j],:])[1]
			end
		end
		return distance_matrix
	end

	"""
		meshgrid(x, y)

	Create a meshgrid from x-min:steps:x-max & y-min:steps:y-max
	"""
	function meshgrid(X, steps=0.1, buffer=1)
		x_min = minimum(X[:,[1]]) - buffer
		y_min = minimum(X[:,[2]]) - buffer
		x_max = maximum(X[:,[1]]) + buffer
		y_max = maximum(X[:,[2]]) + buffer

		x = collect(x_min:steps:x_max)
		y = collect(y_min:steps:y_max)

		X = vec([i for i in x, j in 1:length(y)])
		Y = vec([j for i in 1:length(x), j in y])
		return x, y, hcat(X,Y)
	end

	"""
		KNN(k, X, y, Xtest)

	K nearest neighbors
	Arguments
	---------
	k: int using k nearest neighbors.
	X: the training set
	y: the classes
	Xtest: the test set which we want to predict the classes

	Returns
	-------
	ypred: predicted classes for Xtest

	"""
	function KNN(K, X, y, Xtest)
		N, D = size(X)
		M, _ = size(Xtest)
		num_classes = length(unique(y))

		# 1. Compute distance with all flowers
		distance_grid = zeros(Float64, (N, M))
		distance_grid = pairwise_distance_matrix(X, Xtest)

		# 2. Find indices for the k closest flowers
		index_grid = mapslices(sortperm, distance_grid', dims=2)[:,1:K]

		# 3. Vote for the major class
		ypred = zeros(Float64, (M, num_classes))

		for m in 1:M
			klasses = y[index_grid[m,:]]
			for k in unique(klasses)
				ypred[m, k] = length(klasses[klasses .== k]) / K
			end
		end

		return mapslices(argmax, ypred, dims=2)
	end
end

# ╔═╡ 94413ee0-fc18-11ea-0212-19853b58d1b5
# Load the data
begin
	iris = dataset("datasets", "iris")
	X = convert(Matrix, iris[:,[:SepalLength,:SepalWidth]])
	Y = recode(iris.Species, "setosa"=> 1, "versicolor"=>2, "virginica"=>3)
	scatter(iris.SepalLength, iris.SepalWidth,group=iris.Species)
end

# ╔═╡ b260bb80-fc18-11ea-3758-b3b70d518b33
# calculate the decision boundaries for K-NN
begin
	# Initials KNN parameters
	K = 3
	xx, yy, Xtest = meshgrid(X, 0.1, 1)

	# predict the species for every gridcell
	ypred = KNN(K, X, Y, Xtest)
	
		# reshape the vector to a matrix and plot the result
	prediction_grid = reshape(ypred,length(xx),length(yy))
	heatmap(xx, yy, prediction_grid', color=["orange","green","purple"], legend=false, 		title="KNN decision boundary K=$K")
	scatter!(iris.SepalLength, iris.SepalWidth,group=iris.Species, legend=true)
end

# ╔═╡ Cell order:
# ╠═6a397f82-fc14-11ea-0174-91941427c27f
# ╠═790d6d50-fc14-11ea-3544-271bd7ec8d95
# ╠═94413ee0-fc18-11ea-0212-19853b58d1b5
# ╠═b260bb80-fc18-11ea-3758-b3b70d518b33

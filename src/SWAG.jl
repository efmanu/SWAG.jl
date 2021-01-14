module SWAG

using Flux

export swag, swa, cyclic_LR


"""
	swag(model, data, L, opt;T = 10, c=1, K=3)

This is used to generate SWAG
# Inputs
		model 	: Flux model 
		data 	: Dataset
		L    	: Loss function
		opt 	: Optimizer
# Keyword Arguments
		T		: Number of steps
		c 		: Moment update frequency
		K 		: maximum number of columns in deviation matrix

"""
function swag(model, data, L, opt;T = 10, c=1, K=3)
	training_loss = 0.0
	ps = Flux.params(model)
	θ_0, re = Flux.destructure(model)	
	all_len = length(θ_0)
	θ_cap = θ_0
	θ_sq_cap = θ_0.^2
	D_cap = Array{Float64}(undef,0) #deviation matrix
	for i in 1:T
		for d in data
			gs = gradient(ps) do
				training_loss = L(d...)
				return training_loss
			end
			Flux.update!(opt, ps, gs)
		end
		(θ_cap, θ_sq_cap, D_cap) = moment_update(ps, i, c, K, θ_cap, θ_sq_cap, D_cap, all_len)

	end
	ϵ_diag = θ_sq_cap - θ_cap.^2
	return D_cap, θ_cap, ϵ_diag

end
"""
	swa(model, data, L;T = 10, c=1, K=3, α_1 = 1.0, α_2 = 0.0)
# Inputs
		model 	: Flux model 
		data 	: Dataset
		L    	: Loss function
		opt 	: Optimizer
# Keyword Arguments
		T		: Number of steps
		c 		: Moment update frequency
		K 		: maximum number of columns in deviation matrix


"""
function swa(model, data, L;T = 10, c=1, K=3, α_1 = 1.0, α_2 = 0.0)
	training_loss = 0.0
	ps = Flux.params(model)
	θ_0, re = Flux.destructure(model)	
	all_len = length(θ_0)
	θ_cap = θ_0
	θ_sq_cap = θ_0.^2
	D_cap = Array{Float64}(undef,0) #deviation matrix
	for i in 1:T
		opt.eta	 = cyclic_LR(i, c, α_1, α_2)
		for d in data
			gs = gradient(ps) do
				training_loss = L(d...)
				return training_loss
			end
			Flux.update!(opt, ps, gs)
		end
		θ_cap = θ_cap_update()
		(θ_cap, θ_sq_cap, D_cap) = moment_update(ps, i, c, K, θ_cap, θ_sq_cap, D_cap, all_len)
	end
	return θ_cap, re(θ_cap)
end

"""
	cyclic_LR(i, c; α_1 = 0.99, α_2 = 0.0001)
To generate cyclic learning rate

# Inputs:
	i 	: Epoch
	c 	: Moment update frequency
	α_1	: Maximum learning rate
	α_2	: Minimum Learning rate

"""
function cyclic_LR(i, c; α_1 = 0.99, α_2 = 0.0001)
	t = (1/c)*(mod(i-1,c)+1)
	α_i = (1 - t)*α_1 + t*α_2
	return α_i
end

function moment_update(ps, i, c, K, θ_cap, θ_sq_cap, D_cap, all_len)
	θ_i = Array{Float64}(undef,0)
	[append!(θ_i, reshape(ps.order.data[j],:,1)) for j in 1:ps.params.dict.count];
	if mod(i,c) == 0
		n = i/c
		#moments
		θ_cap = (n.*θ_cap + θ_i)./(n+1)
		θ_sq_cap = (n.*θ_sq_cap + θ_i.^2)./(n+1)
		if(length(D_cap) >= K*all_len)
			D_cap = D_cap[1:(end - all_len)]
		end
		θ_dev = θ_i - θ_cap #calculate deviation
		append!(D_cap, θ_dev)
	end
	return (θ_cap, θ_sq_cap, D_cap)

end

end # module

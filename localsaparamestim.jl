using DiffEqSensitivity, OrdinaryDiffEq, DiffEqParamEstim,
      ParameterizedFunctions, BlackBoxOptim, NLopt, Calculus, ForwardDiff, Optim, BenchmarkTools, RecursiveArrayTools
    
function myl2loss(oursol_type, data)
    my_l2loss = zero(eltype(data))
    for j in 1:size(data)[2]
        for i in 1:size(data)[1]
            my_l2loss += (oursol_type[i,j] - data[i,j])^2
        end
    end
    my_l2loss
end

costfunc = function (p)
    tmp_prob = remake(ourprob;p=p)
    sol = solve(tmp_prob,Tsit5(),saveat=t,abstol=1e-6,reltol=1e-6)
    loss = myl2loss(sol,data)
    loss
end

function myl2lossgradient!(grad,oursol_type,data,num_p)
    fill!(grad,0.0)
    data_x_size = size(data)[1]
    my_grad = -1 .*2 .*(data .- oursol_type[1:data_x_size,:])
    for k in 1:size(my_grad)[2]
        for i in 1:num_p
            for j in i*data_x_size+1:(i+1)*data_x_size
                grad[i] += my_grad[j-i*data_x_size,k]*oursol_type[j,k]
            end
        end
    end
end

# function myl2lossgradient1!(grad,oursol_type,data,num_p)
#     my_gradient_init = -1 .*2 .*(data .- oursol_type[1:size(data)[1],:])
#     for i in 1:num_p
#         grad[i] = sum(my_gradient_init.*oursol_type[i*size(data)[1]+1:(i+1)*size(data)[1],:])
#     end
# end

function costfunc_gradient(p,grad)
    tmp_prob = remake(ourprob;p=p)
    sol = solve(tmp_prob,Tsit5(),saveat=t,abstol=1e-6,reltol=1e-6)
    grad = myl2lossgradient!(grad,sol,data,length(p))
    costfunc(p)
end
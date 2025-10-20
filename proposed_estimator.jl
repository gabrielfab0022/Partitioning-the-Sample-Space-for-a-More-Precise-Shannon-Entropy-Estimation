#=======================================================================================
Project: [Partitioning the Sample Space for a More Precise Shannon Entropy Estimation]
Purpose: Implementation of the methods described in the paper.

Authors: Gabriel F.A. Bastos and Jugurta Montalvão
=======================================================================================#


using Statistics
using Distributions

#DEFINING USEFUL FUNCTIONS
function draw_from_distribution(p, n) #draw n samples from the distribution p 
    CDF = cumsum(p)
    aux = rand(n)
    x = [findall(CDF.>=aux[i])[1] for i in 1:n]
    return x 
end 

function frequencies(x, p)
    n = length(p)
    freqs = [sum(x.==i) for i in 1:n]
    return freqs 
end 

function total_mass_k(x,p, k) #ground trutb total mass
    n = length(p)
    freqs = frequencies(x,p)
    k_occur = findall(freqs.==k)
    mass_k = sum([p[i] for i in k_occur])
    return mass_k 
end

function histogram_of_counts(x) #profile of the sample x
    n = length(x)
    unique_x = unique(x)
    symbols_counts = [sum(x.==i) for i in unique_x] 
    Phi = [sum(symbols_counts.==i) for i in 1:n]
    return Phi 
end 

function obtain_factorial(x)
    if x == 0 
        return 1 
    else 
        return x*obtain_factorial(x-1)
    end 
end

function factorial_division(n, k) #n!/k!
    if n==k 
        return 1
    else 
        return n*factorial_division(n-1, k)
    end 
end

function binom_coefficient(x,y) 
    return factorial_division(x, x-y)/(obtain_factorial(y))
end

function my_estimate_h(p) #calculate the entropy of a distribution p
    H = -sum(p.*log.(p.+eps()))
    return H 
end

poisson_tail(r, z) = ccdf(Poisson(r), z-1)


#TOTAL MASS ESTIMATOR OF THE SYMBOLS THAT WERE OBSERVED k TIMES (MISSING MASS FOR k=0)
function estimate_Mk(x, k) #estimate the total 
    n = length(x)
    PHI = histogram_of_counts(x)
    Mk = sum([(-1)^i * PHI[k+i]/binomial(BigInt(n),k+i) for i in 1:n-k])
    Mk = -binom_coefficient(n,k)*Mk 
    return Mk 
end 



#ESTIMATING THE NUMBER OF UNSEEN SYMBOLS
function get_s(i, mu, a, r, PHI)
    if PHI[i]>0
        sum_lim = min(i, mu-1)
        #tail = [poisson_tail(r,i+j) for j in 0:sum_lim]
        #aux_vec = [tail[j+1]>1e-10 ? ((-a)^i)*((-1)^j)*binom_coefficient(i,j)*tail[j+1] : 0 for j in 0:sum_lim]
        aux_vec = [((-a)^i)*((-1)^j)*binom_coefficient(i,j)*poisson_tail(r,i+j) for j in 0:sum_lim]
        aux_vec = -1*aux_vec
        return sum(aux_vec)
    else 
        return 0
    end 
end

function estimate_u(x, mu, a, r)
    if a>=1 
        n = length(x)
        s = zeros(n) 
        PHI = histogram_of_counts(x)
        last_nonzero = findall(PHI.>0)[end]
        s_aux = [get_s(i, mu, a, r, PHI) for i in 1:last_nonzero]
        s[1:last_nonzero] = s_aux
        U = s'*PHI
        return U 
    else 
        return 0
    end
end

############## PROPOSED ESTIMATOR #############

#ESTIMATING THE DISTRIBUTION Pr(X in S1), Pr(X in S2) and Pr(X in S3)
function estimate_Pg(x, threshold)
    n = length(x)
    Pg = zeros(3)
    Pg[1] = Float64(estimate_Mk(x, 0))
    if Pg[1] < 0
        Pg[1] = 0
    end
    Pg[2] = Float64(sum([estimate_Mk(x, i) for i in 1:threshold]))
    if Pg[2] < 0
        Pg[2] = 0
    end
    Pg[3] = 1-Pg[1]-Pg[2]
    if Pg[3] < 0
        Pg[3] = 0
    end
    Pg = Pg/sum(Pg);
    return Pg 
end


#ESTIMATING THE SECOND TERM 
function estimate_Hg1(x, a)
    n = length(x)
    if a>=1 
        r = log(n*(a+1)^2/(a-1))/(2*a)
    else 
        r = 0
    end
    unseen = Int(round(estimate_u(x, 1, a, r)))
    if unseen <= 0
        return 0 
    else 
        return log(unseen)
    end 
end 

#ESTIMATING THE THIRD TERM 
function estimate_Hg2(x, limiar)
    
    observed_symbols = unique(x)
    freqs = [sum(x.==i) for i in observed_symbols]
    
    universe = observed_symbols[findall(freqs.>0 .&& freqs.<=limiar)]
    if length(universe)>0
        Hg2 = log(length(universe))
    else 
        Hg2 = 0
    end
    return Hg2
end

#ESTIMATING THE LAST TERM 
function estimate_Hg3(x, threshold)
    observed_symbols = unique(x)
    freqs = [sum(x.==i) for i in observed_symbols]
    freqs_filtered = freqs[freqs.>threshold]
    if sum(freqs_filtered)>0
        P = freqs_filtered/sum(freqs_filtered)
        Hg3 = my_estimate_h(P)
        Hg3 = Hg3 + (length(freqs_filtered)-1)/(2*sum(freqs_filtered)) #MILLER-MADOW BIAS CORRECTION
        return Hg3
    else
        return 0 
    end 
end

function get_optimal_a(m) #LOOK-UP TABLE
    if m > 0.8
        return 400000
    elseif m>0.7
        return 100
    elseif m>0.55
        return 8
    elseif m>0.4
        return 5
    elseif m>0.3 
        return 2
    elseif m>0.15
        return 1.5
    else 
        return 1
    end
end


#GETTING TEST DISTRIBUTIONS


#S = 500
S = 1000 #alphabet size
#S = 5000

####GENERATING DISTRIBUTIONS 
P1 = ones(S)/S #UNIFORM
alfa = ones(S)*0.2; d = Dirichlet(alfa); P2 = rand(d); #Dirichlet prior 1/2
alfa = ones(S)*0.05; d = Dirichlet(alfa); P3 = rand(d); #Dirichlet prior 0.05
alfa = ones(S)*0.03; d = Dirichlet(alfa); P4 = rand(d); #Dirichlet prior 1/2
alfa = 2; P5 = collect(1:1:S).^(-alfa+eps()); P5 = P5/sum(P5) #Zipf law, alpha = 2
alfa = 1; P6 = collect(1.0:1.0:S).^(-alfa); P6 = P6/sum(P6) #Zipf law, alpha = 1
alfa = 0.5; P7 = collect(1:1:S).^(-alfa); P7 = P7/sum(P7) #Zipf law, alpha = 0.5


H_P1 = my_estimate_h(P1)
H_P2 = my_estimate_h(P2)
H_P3 = my_estimate_h(P3)
H_P4 = my_estimate_h(P4)
H_P5 = my_estimate_h(P5)
H_P6 = my_estimate_h(P6)
H_P7 = my_estimate_h(P7)
println("GROUND TRUTH")
println("P1: ", H_P1)
println("P2: ", H_P2)
println("P3: ", H_P3)
println("P4: ", H_P4)
println("P5: ", H_P5)
println("P6: ", H_P6)
println("P7: ", H_P7)

##############TESTS#################

#n_samples = [50, 100, 150, 250, 500, 1000, 2500] #FOR S=500
n_samples = [100, 200, 300, 500, 1000, 2000, 5000] #FOR S=1000
#n_samples = [500, 1000, 2000] #FOR S=5000
n_runs = 1000

estimations_p1 = zeros(n_runs, length(n_samples))
estimations_p2 = zeros(n_runs, length(n_samples))
estimations_p3 = zeros(n_runs, length(n_samples))
estimations_p4 = zeros(n_runs, length(n_samples))
estimations_p5 = zeros(n_runs, length(n_samples))
estimations_p6 = zeros(n_runs, length(n_samples))
estimations_p7 = zeros(n_runs, length(n_samples))



for n in 1:length(n_samples)
    println(n)
    @threads for run in 1:n_runs
        #P1#
        x = draw_from_distribution(P1, n_samples[n])
        Mk = estimate_Mk(x,0)
        a = get_optimal_a(Mk)
        threshold = 3
        Pg = estimate_Pg(x, threshold); global estimations_p1[run, n] = my_estimate_h(Pg) + Pg[1]*estimate_Hg1(x, a) + Pg[2]*estimate_Hg2(x, threshold) + Pg[3]*estimate_Hg3(x, threshold)
        
        #P2#
        x = draw_from_distribution(P2, n_samples[n])
        Mk = estimate_Mk(x,0)
        a = get_optimal_a(Mk)
        threshold = 3
        Pg = estimate_Pg(x, threshold); global estimations_p2[run, n] = my_estimate_h(Pg) + Pg[1]*estimate_Hg1(x, a) + Pg[2]*estimate_Hg2(x, threshold) + Pg[3]*estimate_Hg3(x, threshold)

        #P3#
        x = draw_from_distribution(P3, n_samples[n])
        Mk = estimate_Mk(x,0)
        a = get_optimal_a(Mk)
        threshold = 3
        Pg = estimate_Pg(x, threshold); global estimations_p3[run, n] = my_estimate_h(Pg) + Pg[1]*estimate_Hg1(x, a) + Pg[2]*estimate_Hg2(x, threshold) + Pg[3]*estimate_Hg3(x, threshold)

        #P4#
        x = draw_from_distribution(P4, n_samples[n])
        Mk = estimate_Mk(x,0)
        a = get_optimal_a(Mk)
        threshold = 3
        Pg = estimate_Pg(x, threshold); global estimations_p4[run, n] = my_estimate_h(Pg) + Pg[1]*estimate_Hg1(x, a) + Pg[2]*estimate_Hg2(x, threshold) + Pg[3]*estimate_Hg3(x, threshold)

        #P5#
        x = draw_from_distribution(P5, n_samples[n])
        Mk = estimate_Mk(x,0)
        a = get_optimal_a(Mk)
        threshold = 3
        Pg = estimate_Pg(x, threshold); global estimations_p5[run, n] = my_estimate_h(Pg) + Pg[1]*estimate_Hg1(x, a) + Pg[2]*estimate_Hg2(x, threshold) + Pg[3]*estimate_Hg3(x, threshold)

        #P6#
        x = draw_from_distribution(P6, n_samples[n])
        Mk = estimate_Mk(x,0)
        a = get_optimal_a(Mk)
        threshold = 3
        Pg = estimate_Pg(x, threshold); global estimations_p6[run, n] = my_estimate_h(Pg) + Pg[1]*estimate_Hg1(x, a) + Pg[2]*estimate_Hg2(x, threshold) + Pg[3]*estimate_Hg3(x, threshold)

        #P7#
        x = draw_from_distribution(P7, n_samples[n])
        Mk = estimate_Mk(x,0)
        a = get_optimal_a(Mk)
        threshold = 3
        Pg = estimate_Pg(x, threshold); global estimations_p7[run, n] = my_estimate_h(Pg) + Pg[1]*estimate_Hg1(x, a) + Pg[2]*estimate_Hg2(x, threshold) + Pg[3]*estimate_Hg3(x, threshold)
    end
end

######PRINTING RESULTS##############

distributions = [
    ("P1 DISTRIBUTION", estimations_p1, H_P1),
    ("P2 DISTRIBUTION", estimations_p2, H_P2),
    ("P3 DISTRIBUTION", estimations_p3, H_P3),
    ("P4 DISTRIBUTION", estimations_p4, H_P4),
    ("P5 DISTRIBUTION", estimations_p5, H_P5),
    ("P6 DISTRIBUTION", estimations_p6, H_P6),
    ("P7 DISTRIBUTION", estimations_p7, H_P7),
]

for (title, estimations, H) in distributions
    println("\n" * "="^60)
    println(title)
    println("="^60)
    for (i, n) in enumerate(n_samples)
        bias = mean(estimations[:, i] .- H)
        rmse = sqrt(mean((estimations[:, i] .- H).^2))
        println("  $(n) samples → Bias: $(round(bias, digits=4)), RMSE: $(round(rmse, digits=4))")
    end
end
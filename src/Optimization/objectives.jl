export least_squares

function least_squares(Dpred,Dtrue)
    r = Dpred-Dtrue
    f = 0.5*vecnorm(r)^2
    g = r
    h = ones(eltype(Dpred),size(Dpred))
    return (f,g,h)
end

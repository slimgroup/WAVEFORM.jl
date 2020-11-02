export least_squares

function least_squares(Dpred,Dtrue)
    r = Dpred-Dtrue
    f = 0.5*norm(r, 2)^2
    g = r
    h = ones(eltype(Dpred),size(Dpred))
    return (f,g,h)
end

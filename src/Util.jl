module Util
export δ

function δ(i::Number, j::Number)::Integer
    if i == j
        return 1
    else
        return 0
    end
end

end
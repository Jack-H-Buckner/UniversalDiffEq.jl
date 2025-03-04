function save_model_parameters(model,file)
    open(file, "w") do file
        write(file,json(model.parameters.process_model))
    end
end

function load_model_parameters!(model,file)
    params = parsefile(file)

    model.parameters.process_model .= params
end
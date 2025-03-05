function save_parameters(model,path)
    CSV.write(path,DataFrame(parameters = model.process_model.parameters[1:end]))
end 

function load_parameters!(model,path)
    params = CSV.read(path)
    model.parameters.process_model[1:end] .=  params.parameters
end 
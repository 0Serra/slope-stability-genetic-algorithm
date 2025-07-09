
def fitness_fellenius(chromosome, ga_parameters, soil_parameters):
    normalized_individual = ga_parameters.normalize_chromosome(
        chromosome)

    soil_parameters.set_circle_surface(
        normalized_individual[0], normalized_individual[1])

    intersections = soil_parameters.find_intersections_slope_and_circle(
        soil_parameters.slope_segments, soil_parameters.circle_equation)

    _, _, _, _, slice_area, slice_center_angle, slice_base_length = soil_parameters.define_slice_properties(
        intersections, soil_parameters.slope_segments, soil_parameters.circle_equation)

    viability = True

    return viability, soil_parameters.fellenius_safety_factor(
        slice_area, slice_center_angle, slice_base_length)

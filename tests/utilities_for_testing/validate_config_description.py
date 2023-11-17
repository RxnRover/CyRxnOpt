def validate_config_description(test_case, config_desc):
    # A config description should be a list of descriptions for each config
    # option of the optimizer
    test_case.assertEqual(type(config_desc), list)

    for description in config_desc:
        # Each config description is a dictionary with relevant information
        test_case.assertEqual(type(description), dict)

        # A description is complete by providing fields for the config option
        # "name", "type", and "value".
        test_case.assertIn("name", description)
        test_case.assertIn("type", description)
        test_case.assertIn("value", description)

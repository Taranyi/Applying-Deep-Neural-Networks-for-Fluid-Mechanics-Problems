
set(SIMULATION_LINK_LIBRARIES SPlisHSPlasH ${PBD_LIBRARIES} PBD_includes)
set(SIMULATION_DEPENDENCIES SPlisHSPlasH Ext_PBD CopySPlisHSPlasHShaders CopyPBDShaders CopyEmitterModels)



set(PBDWRAPPER_SOURCE_FILES	
	PositionBasedDynamicsWrapper/PBDWrapper.cpp
	PositionBasedDynamicsWrapper/PBDBoundarySimulator.cpp
)
set(PBDWRAPPER_HEADER_FILES
	PositionBasedDynamicsWrapper/PBDWrapper.h
	PositionBasedDynamicsWrapper/PBDRigidBody.h
	PositionBasedDynamicsWrapper/PBDBoundarySimulator.h
)

set(EXPORTER_SOURCE_FILES
	${PROJECT_SOURCE_DIR}/Simulator/Exporter/ParticleExporter_Partio.cpp
	${PROJECT_SOURCE_DIR}/Simulator/Exporter/ParticleExporter_VTK.cpp
	${PROJECT_SOURCE_DIR}/Simulator/Exporter/RigidBodyExporter_BIN.cpp
	${PROJECT_SOURCE_DIR}/Simulator/Exporter/RigidBodyExporter_OBJ.cpp
	${PROJECT_SOURCE_DIR}/Simulator/Exporter/RigidBodyExporter_VTK.cpp
)

set(EXPORTER_HEADER_FILES
	${PROJECT_SOURCE_DIR}/Simulator/Exporter/ExporterBase.h
	${PROJECT_SOURCE_DIR}/Simulator/Exporter/ParticleExporter_Partio.h
	${PROJECT_SOURCE_DIR}/Simulator/Exporter/ParticleExporter_VTK.h
	${PROJECT_SOURCE_DIR}/Simulator/Exporter/RigidBodyExporter_BIN.h
	${PROJECT_SOURCE_DIR}/Simulator/Exporter/RigidBodyExporter_OBJ.h
	${PROJECT_SOURCE_DIR}/Simulator/Exporter/RigidBodyExporter_VTK.h
)



add_library(SimulatorBase
	${PROJECT_SOURCE_DIR}/Simulator/ExporterRegistration.cpp
	${PROJECT_SOURCE_DIR}/Simulator/SceneConfiguration.cpp
	${PROJECT_SOURCE_DIR}/Simulator/SceneConfiguration.h
	${PROJECT_SOURCE_DIR}/Simulator/SimulatorBase.cpp
	${PROJECT_SOURCE_DIR}/Simulator/SimulatorBase.h
	${PROJECT_SOURCE_DIR}/Simulator/BoundarySimulator.cpp
	${PROJECT_SOURCE_DIR}/Simulator/BoundarySimulator.h
	${PROJECT_SOURCE_DIR}/Simulator/StaticBoundarySimulator.cpp
	${PROJECT_SOURCE_DIR}/Simulator/StaticBoundarySimulator.h
	
	${EXPORTER_SOURCE_FILES}
	${EXPORTER_HEADER_FILES}

	${PBDWRAPPER_SOURCE_FILES}
	${PBDWRAPPER_HEADER_FILES}
	
)

if(DL_OUTPUT)
	target_compile_definitions(SimulatorBase PUBLIC -DDL_OUTPUT)
endif()




set_target_properties(SimulatorBase PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(SimulatorBase PROPERTIES RELWITHDEBINFO_POSTFIX ${CMAKE_RELWITHDEBINFO_POSTFIX})
set_target_properties(SimulatorBase PROPERTIES MINSIZEREL_POSTFIX ${CMAKE_MINSIZEREL_POSTFIX})
add_dependencies(SimulatorBase ${SIMULATION_DEPENDENCIES})
target_link_libraries(SimulatorBase PUBLIC ${SIMULATION_LINK_LIBRARIES})


source_group("Header Files\\Exporter" FILES ${EXPORTER_HEADER_FILES})
source_group("Source Files\\Exporter" FILES ${EXPORTER_SOURCE_FILES})
source_group("Header Files\\PBD" FILES ${PBDWRAPPER_HEADER_FILES})
source_group("Source Files\\PBD" FILES ${PBDWRAPPER_SOURCE_FILES})

add_executable(SPHSimulator main.cpp)
target_link_libraries(SPHSimulator SimulatorBase)
add_dependencies(SPHSimulator SimulatorBase)

set_target_properties(SPHSimulator PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(SPHSimulator PROPERTIES RELWITHDEBINFO_POSTFIX ${CMAKE_RELWITHDEBINFO_POSTFIX})
set_target_properties(SPHSimulator PROPERTIES MINSIZEREL_POSTFIX ${CMAKE_MINSIZEREL_POSTFIX})


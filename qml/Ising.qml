import QtQuick 2.1
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import org.julialang 1.0
import Qt.labs.platform 1.1


ApplicationWindow {
  title: "Ising Simulation"
  width: 800
  height: 800
  visible: true



  // Whole application in a column
  // Screen stuff above, panels and text under
  ColumnLayout{
    // anchors.fill: parent
    anchors.centerIn: parent
    spacing: 32

    // Buttons at top of window
    Item{
      Layout.alignment: Qt.AlignHCenter
      width: childrenRect.width
      height: childrenRect.height

      ColumnLayout{
        spacing: 2
        // Init Graph Button
        Button {
          Layout.alignment: Qt.AlignCenter
          text: "Initialize Graph"
          onClicked: Julia.initIsing()
        }
        // Pause Simulation Button
        Button{
          Layout.alignment: Qt.AlignCenter
          text: {
            if(obs.isPaused)
            {
              "Paused"
            }
            else{
              "Running"
            }
          }
          onClicked: {
            obs.isPaused = !obs.isPaused
          }
        }
      }
    }


    // Middle row
    // Brush - Screen - Temperature
    RowLayout{
      spacing: 6
      Layout.alignment: Qt.AlignHCenter
      // Brush Buttons and radius slider
      RowLayout{

        ColumnLayout{
          Text{
            Layout.alignment: Qt.AlignHCenter
            text: "Type"
          }
          Layout.alignment: Qt.AlignCenter

          Button{
            id: brushButton
            text: "1"
            onClicked: {
              obs.brush = 1
            }
          }
          Button{
            Layout.preferredWidth: brushButton.width
            text: "0"
            onClicked: {
              obs.brush = 0
            }
          }
          Button{
            Layout.preferredWidth: brushButton.width
            text: "-1"
            onClicked: {
              obs.brush = -1
            }
          }

        }
        // Radius Slider
        ColumnLayout{
          Text{
            text: "Radius \n" + obs.brushR
          }
          Slider{
            value: obs.brushR
            orientation: Qt.Vertical
            minimumValue: 1
            maximumValue: 100
            stepSize: 1
            onValueChanged: {
              obs.brushR = value
            }
          }
        }
      }

      JuliaDisplay {
        Layout.alignment: Qt.AlignCenter
        id: jdisp
        width: 512
        height: 512

        Component.onCompleted: {
          Julia.initIsing()
          // Julia.updateIsing(jdisp)
          Julia.persistentFunctions(jdisp)
        }

        MouseArea{
          anchors.fill: parent
          onClicked: {
            Julia.circleToStateQML(mouseY, mouseX)
          }
        }





      }
      //  Temperature Slider
      Item{
        Layout.alignment: Qt.AlignCenter
        width: childrenRect.width
        height: childrenRect.height
        RowLayout{
          spacing: 2
          // Temp Slider
          Slider{
            value: obs.TIs
            orientation: Qt.Vertical
            minimumValue: 0.0
            maximumValue: 20
            stepSize: 0.01
            onValueChanged: {
              obs.TIs = value
            }
          }
          // Temperature text
          Item{
            width: 32
            Text{
              Layout.alignment: Qt.AlignCenter
              text: qsTr("T=\n") + obs.TIs.toFixed(2)
            }
          }
        }
      }
    }

    // Panels under plot
    // Inserting Defects
    // Magnetization
    ColumnLayout{
      Layout.alignment: Qt.AlignCenter
      spacing: 2

      // Defects textfield & Button
      Item{
        Layout.alignment: Qt.AlignCenter

        width: childrenRect.width
        height: childrenRect.height

        ColumnLayout{
          spacing: 2

          TextField{
            Layout.alignment: Qt.AlignHCenter

            text: obs.pDefects
            onTextChanged: {
              obs.pDefects = parseInt(text)
              if(obs.pDefects > 100)
              {
                obs.pDefects = 100
              }
            }
          }
          Button{
            Layout.alignment: Qt.AlignHCenter
            text: "Insert Defects"
            onClicked: {
              // obs.routine = "insDefects"
              // obs.specialRoutine = true
              Julia.addRandomDefectsQML()
            }
          }
        }
      }

      // Text under plot, magnetization
      Item{
        width: childrenRect.width
        height: childrenRect.height
        Layout.alignment: Qt.AlignCenter
        ColumnLayout{

          spacing: 2
          Text{
            text: "Magnetization: " + obs.M.toFixed(1)
          }
          // Text{
          //   text: "Updates per frame: " + obs.uframe
          // }
        }
      }
      
      // Initiate Temperature sweep
      Button{
        text: "Temperature Sweep"
        onClicked:{
          Julia.tempSweepQML()
        }
      }
      
      // Doesn't Work
      // Button{
      //   Layout.alignment: Qt.AlignCenter
      //   text: "Quit"
      //   onClicked: {
      //     obs.running = false
      //     Qt.quit()
      //   }
      // }

    }
  }
}


// Button{
//   Layout.alignment: Qt.AlignCenter
//   text: "Print g"
//   onClicked: {
//     Julia.printG()
//   }
// }

// Item{
//   id: BrushButtons
//   RowLayout{
//     Layout.alignment: Qt.AlignCenter
//     ColumnLayout{
//       Text{
//         text: "Circle Type"
//       }
//       Button{
//         text: "1"
//         onClicked: {
//           obs.brush = 1
//         }
//       }
//       Button{
//         text: "0"
//         onClicked: {
//           obs.brush = 0
//         }
//       }
//       Button{
//         text: "-1"
//         onClicked: {
//           obs.brush = -1
//         }
//       }

//     }
//     // Radius Slider
//     ColumnLayout{
//       Text{
//         text: "Radius"
//       }
//       Slider{
//         value: obs.brushR
//         orientation: Qt.Vertical
//         minimumValue: 1
//         maximumValue: 100
//         stepSize: 1
//       onValueChanged: {
//         obs.brushR = value
//       }
//       }
//     }
//   }

// }

// MenuBar {
//   id: menuBar
//   Menu {
//     id: fileMenu
//     title: qsTr("File")
//     Action {
//       text: qsTr("&New...")
//     }
//     Action {
//       text: qsTr("&Open...")
//     }
//     Action {
//       text: qsTr("&Save")
//     }
//     Action {
//       text: qsTr("Save &As...")
//     }
//     MenuSeparator { }
//     Action {
//       text: qsTr("&Quit")
//     }
//   }

//   Menu {
//     id: editMenu
//     title: qsTr("&Edit")
//     // ...
//   }

//   Menu {
//     id: viewMenu
//     title: qsTr("&View")
//     // ...
//   }

//   Menu {
//     id: helpMenu
//     title: qsTr("&Help")
//     // ...
//   }
// }
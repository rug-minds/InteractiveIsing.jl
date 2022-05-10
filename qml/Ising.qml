import QtQuick 2.1
import QtQuick.Controls 1.0
import QtQuick.Layouts 1.0
import org.julialang 1.0

ApplicationWindow {
  title: "Ising Simulation"
  width: 800
  height: 800
  visible: true


  
  ColumnLayout{
  anchors.centerIn: parent
  spacing: 32
      // Brush Buttons and radius slider
      
      RowLayout{
        spacing: 2

          RowLayout{
      Layout.alignment: Qt.AlignCenter
      ColumnLayout{
        Text{
          text: "Circle Type"
        }
        Button{
          text: "1"
          onClicked:{
            obs.brush = 1
          }
        }
        Button{
          text: "0"
          onClicked:{
            obs.brush = 0
          }
        }
        Button{
          text: "-1"
          onClicked:{
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
      onValueChanged:{
        obs.brushR = value
      }
      }
    }
  }

        ColumnLayout {
          spacing: 6
          

          // TextField{
          //   Layout.alignment: Qt.AlignCenter
          //   text: Lattice Size
          //   onAccepted: observables.NIsing = parseInt(text)
          // }

          Button {
            Layout.alignment: Qt.AlignCenter
            text: "Initialize Graph"
            onClicked: Julia.initIsing()
          }

          Button{
            Layout.alignment: Qt.AlignCenter
            text: {
              if(obs.isPaused){
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
          
          // Display
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
              onClicked:{
                Julia.circleToStateQML(mouseY,mouseX)
              }
            }

          
          }
        
      }
      //  Temperature Slider
        Slider{
          value: obs.TIs
          orientation: Qt.Vertical
          minimumValue: 0.0
          maximumValue: 30
          stepSize: 0.1
          onValueChanged:{
            obs.TIs = value
          }
        }
        // Temperature number
        Item{
          width: 32
          Text{
            Layout.alignment: Qt.AlignCenter
            text: qsTr("T=\n") + obs.TIs.toFixed(1)
          }
        }
        

      }
      // Under plot and slider
      ColumnLayout{
        Layout.alignment: Qt.AlignCenter

        TextField{
          Layout.alignment: Qt.AlignCenter
          text: obs.pDefects
          onTextChanged:{
            obs.pDefects = parseInt(text)
            if(obs.pDefects > 100){obs.pDefects = 100}
          }
        }
        Button{
          Layout.alignment: Qt.AlignCenter
          text: "Insert Defects"
          onClicked: {
            // obs.routine = "insDefects"
            // obs.specialRoutine = true
            Julia.addRandomDefectsQML()
          }
        }

        Button{
          Layout.alignment: Qt.AlignCenter
          text: "Print g"
          onClicked: {
            Julia.printG()
          }

        }
      }
      
    }
}


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
//         onClicked:{
//           obs.brush = 1
//         }
//       }
//       Button{
//         text: "0"
//         onClicked:{
//           obs.brush = 0
//         }
//       }
//       Button{
//         text: "-1"
//         onClicked:{
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
//       onValueChanged:{
//         obs.brushR = value
//       }
//       }
//     }
//   }

// }